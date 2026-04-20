"""Tests for the benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path

from godspeed.training.benchmark import (
    BenchmarkResult,
    BenchmarkScore,
    BenchmarkTask,
    _jaccard_similarity,
    _lcs_length,
    _lcs_ratio,
    aggregate_scores,
    load_tasks,
    score_result,
)

# -- Fixtures ---------------------------------------------------------------

SAMPLE_TASK = BenchmarkTask(
    task_id="test-01",
    prompt="Fix the bug",
    expected_tools=["grep_search", "file_read", "file_edit"],
    expected_tool_sequence=["grep_search", "file_read", "file_edit"],
    success_criteria="Bug is fixed",
    difficulty="medium",
)


# -- Scoring helper tests ---------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self) -> None:
        assert _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_empty_sets(self) -> None:
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        assert _jaccard_similarity({"a"}, set()) == 0.0


class TestLCSLength:
    def test_identical(self) -> None:
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_subsequence(self) -> None:
        assert _lcs_length(["a", "b", "c"], ["a", "x", "b", "y", "c"]) == 3

    def test_no_common(self) -> None:
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_empty(self) -> None:
        assert _lcs_length([], ["a"]) == 0

    def test_partial_match(self) -> None:
        assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2


class TestLCSRatio:
    def test_perfect_match(self) -> None:
        assert _lcs_ratio(["a", "b"], ["a", "b"]) == 1.0

    def test_empty_expected(self) -> None:
        assert _lcs_ratio([], []) == 1.0
        assert _lcs_ratio([], ["a"]) == 0.0

    def test_half_match(self) -> None:
        assert _lcs_ratio(["a", "b"], ["a", "c"]) == 0.5


# -- Score result tests ------------------------------------------------------


class TestScoreResult:
    def test_perfect_score(self) -> None:
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["grep_search", "file_read", "file_edit"],
            tool_sequence=["grep_search", "file_read", "file_edit"],
            completed=True,
        )
        score = score_result(SAMPLE_TASK, result)
        assert score.tool_selection == 1.0
        assert score.sequence_quality == 1.0
        assert score.overall == 1.0

    def test_partial_tools(self) -> None:
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["grep_search", "file_read"],
            tool_sequence=["grep_search", "file_read"],
        )
        score = score_result(SAMPLE_TASK, result)
        assert 0 < score.tool_selection < 1.0
        assert 0 < score.sequence_quality < 1.0

    def test_wrong_tools(self) -> None:
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["shell", "git"],
            tool_sequence=["shell", "git"],
        )
        score = score_result(SAMPLE_TASK, result)
        assert score.tool_selection == 0.0
        assert score.sequence_quality == 0.0
        assert score.overall == 0.0

    def test_extra_tools(self) -> None:
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["grep_search", "file_read", "file_edit", "shell"],
            tool_sequence=["grep_search", "file_read", "file_edit", "shell"],
        )
        score = score_result(SAMPLE_TASK, result)
        # All expected present but extra tool reduces Jaccard
        assert score.tool_selection < 1.0
        assert score.sequence_quality == 1.0

    def test_returns_benchmark_score(self) -> None:
        result = BenchmarkResult(task_id="test-01")
        score = score_result(SAMPLE_TASK, result)
        assert isinstance(score, BenchmarkScore)

    def test_waste_penalty_zero_when_efficient(self) -> None:
        """Running exactly the expected tool count incurs no penalty."""
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["grep_search", "file_read", "file_edit"],
            tool_sequence=["grep_search", "file_read", "file_edit"],
        )
        score = score_result(SAMPLE_TASK, result)
        assert score.waste_penalty == 0.0
        assert score.overall == 1.0

    def test_waste_penalty_zero_within_slack(self) -> None:
        """Up to 1.5x expected calls is considered fine — no penalty."""
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["grep_search", "file_read", "file_edit"],
            tool_sequence=[
                "grep_search",
                "file_read",
                "file_read",
                "file_edit",
            ],  # 4/3 ratio, below 1.5x threshold
        )
        score = score_result(SAMPLE_TASK, result)
        assert score.waste_penalty == 0.0

    def test_waste_penalty_kicks_in_when_overrun(self) -> None:
        """More than 1.5x expected calls incurs a small, capped penalty."""
        result = BenchmarkResult(
            task_id="test-01",
            tools_used=["grep_search", "file_read", "file_edit"],
            tool_sequence=[
                "grep_search",
                "file_read",
                "file_read",
                "file_read",
                "file_read",
                "file_read",
                "file_edit",
            ],  # 7/3 ratio, well over 1.5x
        )
        score = score_result(SAMPLE_TASK, result)
        assert score.waste_penalty > 0.0
        assert score.waste_penalty <= 0.3  # cap enforced
        # Overall should still be clamped to [0, 1]
        assert 0.0 <= score.overall <= 1.0


# -- Aggregate scores tests -------------------------------------------------


class TestAggregateScores:
    def test_basic_aggregation(self) -> None:
        scores = [
            BenchmarkScore("t1", tool_selection=1.0, sequence_quality=1.0, overall=1.0),
            BenchmarkScore("t2", tool_selection=0.5, sequence_quality=0.5, overall=0.5),
        ]
        result = aggregate_scores(scores)
        assert result.total_tasks == 2
        assert result.mean_tool_selection == 0.75
        assert result.mean_sequence_quality == 0.75
        assert result.mean_overall == 0.75

    def test_empty_scores(self) -> None:
        result = aggregate_scores([])
        assert result.total_tasks == 0
        assert result.mean_overall == 0.0


# -- Load tasks tests -------------------------------------------------------


class TestLoadTasks:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_data = [
            {
                "task_id": "t1",
                "prompt": "Do something",
                "expected_tools": ["file_read"],
            },
            {
                "task_id": "t2",
                "prompt": "Do another",
                "expected_tools": ["shell"],
                "difficulty": "hard",
            },
        ]
        with open(tasks_file, "w", encoding="utf-8") as f:
            for t in tasks_data:
                f.write(json.dumps(t) + "\n")

        tasks = load_tasks(tasks_file)
        assert len(tasks) == 2
        assert tasks[0].task_id == "t1"
        assert tasks[0].difficulty == "medium"  # default
        assert tasks[1].difficulty == "hard"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_file.write_text(
            '{"task_id": "t1", "prompt": "ok", "expected_tools": ["a"]}\n'
            "not json\n"
            '{"task_id": "t2", "prompt": "ok", "expected_tools": ["b"]}\n',
            encoding="utf-8",
        )
        tasks = load_tasks(tasks_file)
        assert len(tasks) == 2

    def test_loads_real_benchmark_file(self) -> None:
        tasks_path = Path(__file__).parent.parent.parent / "benchmarks" / "tasks.jsonl"
        if tasks_path.exists():
            tasks = load_tasks(tasks_path)
            assert len(tasks) >= 15
            for task in tasks:
                assert task.task_id
                assert task.prompt
                assert task.expected_tools
