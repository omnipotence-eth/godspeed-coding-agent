"""Tests for the fitness evaluator — A/B testing with LLM-as-judge scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from godspeed.evolution.fitness import (
    FitnessEvaluator,
    FitnessScore,
    JudgeVerdict,
)
from godspeed.evolution.mutator import MutationCandidate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    original: str = "Short description.",
    mutated: str = "Improved longer description with examples.",
    artifact_type: str = "tool_description",
    artifact_id: str = "bash",
) -> MutationCandidate:
    return MutationCandidate(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        original=original,
        mutated=mutated,
        mutation_rationale="test",
        model_used="ollama/test",
    )


GOOD_VERDICT_JSON = """{
    "a_correctness": 6, "a_procedure": 5, "a_conciseness": 7,
    "b_correctness": 9, "b_procedure": 8, "b_conciseness": 8,
    "winner": "b"
}"""

TIE_VERDICT_JSON = """{
    "a_correctness": 7, "a_procedure": 7, "a_conciseness": 7,
    "b_correctness": 7, "b_procedure": 7, "b_conciseness": 7,
    "winner": "tie"
}"""


# ---------------------------------------------------------------------------
# Test: FitnessScore data structure
# ---------------------------------------------------------------------------


class TestFitnessScore:
    def test_frozen(self) -> None:
        score = FitnessScore(
            correctness=0.9,
            procedure_following=0.8,
            conciseness=0.7,
            overall=0.83,
            length_penalty=0.0,
            confidence=1.0,
        )
        with pytest.raises(AttributeError):
            score.overall = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        score = FitnessScore(
            correctness=0.9,
            procedure_following=0.8,
            conciseness=0.7,
            overall=0.83,
            length_penalty=0.1,
            confidence=0.5,
        )
        assert score.correctness == 0.9
        assert score.confidence == 0.5


# ---------------------------------------------------------------------------
# Test: JudgeVerdict data structure
# ---------------------------------------------------------------------------


class TestJudgeVerdict:
    def test_frozen(self) -> None:
        v = JudgeVerdict(
            a_correctness=7,
            a_procedure=6,
            a_conciseness=8,
            b_correctness=9,
            b_procedure=8,
            b_conciseness=7,
            winner="b",
        )
        with pytest.raises(AttributeError):
            v.winner = "a"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: _parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_valid_json(self) -> None:
        verdict = FitnessEvaluator._parse_verdict(GOOD_VERDICT_JSON)
        assert verdict is not None
        assert verdict.winner == "b"
        assert verdict.b_correctness == 9

    def test_json_in_code_block(self) -> None:
        text = f"```json\n{GOOD_VERDICT_JSON}\n```"
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "b"

    def test_json_with_surrounding_text(self) -> None:
        text = f"Here is my evaluation:\n{GOOD_VERDICT_JSON}\nThat's my verdict."
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None

    def test_no_json(self) -> None:
        verdict = FitnessEvaluator._parse_verdict("No JSON here at all")
        assert verdict is None

    def test_missing_keys(self) -> None:
        verdict = FitnessEvaluator._parse_verdict('{"a_correctness": 5}')
        assert verdict is None

    def test_invalid_winner_defaults_to_tie(self) -> None:
        j = (
            '{"a_correctness": 5, "a_procedure": 5,'
            ' "a_conciseness": 5, "b_correctness": 5,'
            ' "b_procedure": 5, "b_conciseness": 5,'
            ' "winner": "invalid"}'
        )
        verdict = FitnessEvaluator._parse_verdict(j)
        assert verdict is not None
        assert verdict.winner == "tie"

    def test_scores_clamped(self) -> None:
        j = (
            '{"a_correctness": -5, "a_procedure": 15,'
            ' "a_conciseness": 5, "b_correctness": 5,'
            ' "b_procedure": 5, "b_conciseness": 5,'
            ' "winner": "a"}'
        )
        verdict = FitnessEvaluator._parse_verdict(j)
        assert verdict is not None
        assert verdict.a_correctness == 0.0
        assert verdict.a_procedure == 10.0


# ---------------------------------------------------------------------------
# Test: evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_successful_evaluation(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = GOOD_VERDICT_JSON

            score = await evaluator.evaluate(candidate, test_cases=["Test task"])

        assert isinstance(score, FitnessScore)
        assert score.correctness == 0.9  # 9/10
        assert score.procedure_following == 0.8  # 8/10
        assert score.conciseness == 0.8  # 8/10
        assert score.confidence == 1.0
        assert score.overall > 0

    @pytest.mark.asyncio
    async def test_multiple_test_cases(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = GOOD_VERDICT_JSON

            score = await evaluator.evaluate(
                candidate,
                test_cases=["Task 1", "Task 2", "Task 3"],
            )

        assert score.confidence == 1.0
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_partial_failures_reduce_confidence(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        call_count = 0

        async def _alternating_response(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("fail")
            return GOOD_VERDICT_JSON

        with patch.object(evaluator, "_call_llm", side_effect=_alternating_response):
            score = await evaluator.evaluate(
                candidate,
                test_cases=["T1", "T2", "T3"],
            )

        # 2 out of 3 succeed (call 1 OK, call 2 fail, call 3 OK)
        assert score.confidence == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_all_failures_return_zero(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("fail")

            score = await evaluator.evaluate(candidate, test_cases=["T1"])

        assert score.overall == 0.0
        assert score.confidence == 0.0

    @pytest.mark.asyncio
    async def test_default_test_case_used(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = GOOD_VERDICT_JSON

            score = await evaluator.evaluate(candidate)  # No test_cases

        assert score.confidence == 1.0
        assert mock_llm.call_count == 1


# ---------------------------------------------------------------------------
# Test: length penalty
# ---------------------------------------------------------------------------


class TestLengthPenalty:
    @pytest.mark.asyncio
    async def test_no_penalty_for_small_growth(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate(
            original="A tool description that is fairly long already.",
            mutated="A tool description that is fairly long already, with a small addition.",
        )

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = GOOD_VERDICT_JSON

            score = await evaluator.evaluate(candidate, test_cases=["T1"])

        assert score.length_penalty == 0.0

    @pytest.mark.asyncio
    async def test_penalty_for_excessive_growth(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate(
            original="short",
            mutated="x" * 1000,  # Way longer than 2x original
        )

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = GOOD_VERDICT_JSON

            score = await evaluator.evaluate(candidate, test_cases=["T1"])

        assert score.length_penalty > 0


# ---------------------------------------------------------------------------
# Test: model configuration
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_default_model_auto_detects(self) -> None:
        with patch("godspeed.evolution.hardware.detect_vram_mb", return_value=14000):
            e = FitnessEvaluator()
            assert e.judge_model == "ollama/gemma3:12b"

    def test_default_model_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware.detect_vram_mb", return_value=4000):
            e = FitnessEvaluator()
            assert e.judge_model == "ollama/qwen2.5:3b"

    def test_custom_model(self) -> None:
        e = FitnessEvaluator(judge_model="anthropic/claude-sonnet-4-20250514")
        assert e.judge_model == "anthropic/claude-sonnet-4-20250514"
