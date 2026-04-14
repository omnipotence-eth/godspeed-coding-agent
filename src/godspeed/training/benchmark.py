"""Benchmark suite for evaluating fine-tuned tool-calling models.

Defines benchmark tasks with expected tool sequences and scoring functions.
Used to measure whether a fine-tuned model improves tool selection accuracy,
sequence quality, and task completion compared to the base model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """A single benchmark task with expected outcomes."""

    task_id: str
    prompt: str
    expected_tools: list[str]
    expected_tool_sequence: list[str]
    success_criteria: str
    difficulty: str  # "easy" | "medium" | "hard"


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """Scoring results for a single benchmark task."""

    task_id: str
    tool_selection: float  # 0.0 to 1.0 — Jaccard similarity
    sequence_quality: float  # 0.0 to 1.0 — LCS / expected length
    overall: float  # weighted average


@dataclass(slots=True)
class BenchmarkResult:
    """Results from running a single benchmark task."""

    task_id: str
    tools_used: list[str] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    completed: bool = False
    error: str | None = None


@dataclass(slots=True)
class BenchmarkSuiteResult:
    """Aggregate results from running the full benchmark suite."""

    total_tasks: int = 0
    completed_tasks: int = 0
    scores: list[BenchmarkScore] = field(default_factory=list)
    mean_tool_selection: float = 0.0
    mean_sequence_quality: float = 0.0
    mean_overall: float = 0.0
    by_difficulty: dict[str, float] = field(default_factory=dict)


def load_tasks(tasks_path: Path) -> list[BenchmarkTask]:
    """Load benchmark tasks from a JSONL file.

    Each line is a JSON object with the BenchmarkTask fields.
    """
    tasks: list[BenchmarkTask] = []
    with open(tasks_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append(
                    BenchmarkTask(
                        task_id=data["task_id"],
                        prompt=data["prompt"],
                        expected_tools=data["expected_tools"],
                        expected_tool_sequence=data.get(
                            "expected_tool_sequence", data["expected_tools"]
                        ),
                        success_criteria=data.get("success_criteria", ""),
                        difficulty=data.get("difficulty", "medium"),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed task at line %d: %s", line_no, exc)
    return tasks


def score_result(task: BenchmarkTask, result: BenchmarkResult) -> BenchmarkScore:
    """Score a benchmark result against the expected task outcome.

    Metrics:
    - tool_selection: Jaccard similarity between expected and actual tool sets
    - sequence_quality: LCS length / expected sequence length
    - overall: 0.6 * tool_selection + 0.4 * sequence_quality
    """
    tool_selection = _jaccard_similarity(set(task.expected_tools), set(result.tools_used))
    sequence_quality = _lcs_ratio(task.expected_tool_sequence, result.tool_sequence)
    overall = round(0.6 * tool_selection + 0.4 * sequence_quality, 3)

    return BenchmarkScore(
        task_id=task.task_id,
        tool_selection=round(tool_selection, 3),
        sequence_quality=round(sequence_quality, 3),
        overall=overall,
    )


def aggregate_scores(scores: list[BenchmarkScore]) -> BenchmarkSuiteResult:
    """Compute aggregate statistics from individual benchmark scores."""
    result = BenchmarkSuiteResult(
        total_tasks=len(scores),
        completed_tasks=len(scores),
        scores=scores,
    )

    if not scores:
        return result

    result.mean_tool_selection = round(sum(s.tool_selection for s in scores) / len(scores), 3)
    result.mean_sequence_quality = round(sum(s.sequence_quality for s in scores) / len(scores), 3)
    result.mean_overall = round(sum(s.overall for s in scores) / len(scores), 3)

    # Group by difficulty (inferred from task_id prefix convention)
    difficulty_scores: dict[str, list[float]] = {}
    for s in scores:
        # Use task_id parts or default
        diff = "unknown"
        for task_score in scores:
            if task_score.task_id == s.task_id:
                diff = s.task_id.split("-")[0] if "-" in s.task_id else "unknown"
                break
        difficulty_scores.setdefault(diff, []).append(s.overall)

    result.by_difficulty = {k: round(sum(v) / len(v), 3) for k, v in difficulty_scores.items()}

    return result


def _jaccard_similarity(expected: set[str], actual: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not expected and not actual:
        return 1.0
    union = expected | actual
    if not union:
        return 0.0
    intersection = expected & actual
    return len(intersection) / len(union)


def _lcs_length(seq_a: list[str], seq_b: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0

    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def _lcs_ratio(expected: list[str], actual: list[str]) -> float:
    """Compute LCS length / expected sequence length."""
    if not expected:
        return 1.0 if not actual else 0.0
    lcs_len = _lcs_length(expected, actual)
    return lcs_len / len(expected)
