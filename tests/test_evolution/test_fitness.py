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
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            e = FitnessEvaluator()
            assert e.judge_model == "ollama/devstral-small-2:24b"

    def test_default_model_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=5500):
            e = FitnessEvaluator()
            assert e.judge_model == "ollama/rnj-1:8b"

    def test_custom_model(self) -> None:
        e = FitnessEvaluator(judge_model="anthropic/claude-sonnet-4-20250514")
        assert e.judge_model == "anthropic/claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Test: _parse_verdict edge cases (expanded)
# ---------------------------------------------------------------------------


class TestParseVerdictEdgeCases:
    def test_json_code_block_with_json_prefix(self) -> None:
        text = (
            '```json\n{"a_correctness":8,"a_procedure":7,'
            '"a_conciseness":6,"b_correctness":9,"b_procedure":8,'
            '"b_conciseness":7,"winner":"b"}\n```'
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.b_correctness == 9

    def test_json_code_block_no_language_tag(self) -> None:
        text = (
            '```\n{"a_correctness":5,"a_procedure":5,'
            '"a_conciseness":5,"b_correctness":5,"b_procedure":5,'
            '"b_conciseness":5,"winner":"tie"}\n```'
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "tie"

    def test_invalid_json_syntax(self) -> None:
        verdict = FitnessEvaluator._parse_verdict("This is not json { broken")
        assert verdict is None

    def test_nested_json_correctly_found(self) -> None:
        text = (
            'Here is my verdict:\n'
            '{"a_correctness":7,"a_procedure":6,'
            '"a_conciseness":8,"b_correctness":7,"b_procedure":6,'
            '"b_conciseness":8,"winner":"a"}'
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "a"

    def test_non_numeric_score_string(self) -> None:
        j = (
            '{"a_correctness":[],"a_procedure":5,'
            '"a_conciseness":5,"b_correctness":5,'
            '"b_procedure":5,"b_conciseness":5,'
            '"winner":"a"}'
        )
        verdict = FitnessEvaluator._parse_verdict(j)
        assert verdict is not None
        assert verdict.a_correctness == 0.0

    def test_non_numeric_score_value_typeerror(self) -> None:
        j = (
            '{"a_correctness":"not_a_number","a_procedure":5,'
            '"a_conciseness":5,"b_correctness":5,'
            '"b_procedure":5,"b_conciseness":5,'
            '"winner":"a"}'
        )
        verdict = FitnessEvaluator._parse_verdict(j)
        assert verdict is not None
        assert verdict.a_correctness == 0.0

    def test_json_decode_error_returns_none(self) -> None:
        text = '{"a_correctness": 5, "b_conciseness": broken, "winner": "a"}'
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is None

    def test_code_block_json_prefix_not_json_content(self) -> None:
        text = (
            '```json\njust some text, no braces\n```\n'
            '{"a_correctness":8,"a_procedure":7,'
            '"a_conciseness":6,"b_correctness":9,"b_procedure":8,'
            '"b_conciseness":7,"winner":"b"}'
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "b"


# ---------------------------------------------------------------------------
# Test: _call_llm branches
# ---------------------------------------------------------------------------


class TestCallLlmBranches:
    @pytest.mark.asyncio
    async def test_call_with_injected_client(self) -> None:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = GOOD_VERDICT_JSON
        mock_client.chat.return_value = mock_response

        evaluator = FitnessEvaluator(llm_client=mock_client)
        result = await evaluator._call_llm("test prompt")
        assert result == GOOD_VERDICT_JSON
        mock_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_without_injected_client_creates_client(self) -> None:
        evaluator = FitnessEvaluator()
        with patch("godspeed.llm.client.LLMClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.content = "response text"
            mock_client.chat.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await evaluator._call_llm("test prompt")
            assert result == "response text"
            mock_client_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _judge exception and _parse_verdict failure
# ---------------------------------------------------------------------------


class TestJudgeEdgeCases:
    @pytest.mark.asyncio
    async def test_judge_parse_failure_returns_none(self) -> None:
        evaluator = FitnessEvaluator()
        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "no json at all"
            result = await evaluator._judge(
                purpose="test",
                original="original text",
                mutated="mutated text",
            )
            assert result is None


# ---------------------------------------------------------------------------
# Test: Evaluate with exception in judge_case (the except branch at lines 126-133)
# ---------------------------------------------------------------------------


class TestEvaluateExceptionHandling:
    @pytest.mark.asyncio
    async def test_judge_case_exception_returns_none(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        async def _failing_call(prompt: str) -> str:
            raise RuntimeError("LLM crash")

        with patch.object(evaluator, "_call_llm", side_effect=_failing_call):
            score = await evaluator.evaluate(candidate, test_cases=["T1"])
            assert score.overall == 0.0
            assert score.confidence == 0.0

    @pytest.mark.asyncio
    async def test_empty_test_cases_defaults_to_one(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = GOOD_VERDICT_JSON

            score = await evaluator.evaluate(candidate, test_cases=[])
            assert score.confidence == 1.0
            assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_judge_case_direct_exception(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        with patch.object(
            evaluator, "_judge", new_callable=AsyncMock, side_effect=RuntimeError("crash before try")
        ):
            score = await evaluator.evaluate(candidate, test_cases=["T1", "T2"])
            assert score.overall == 0.0
            assert score.confidence == 0.0


# ---------------------------------------------------------------------------
# Test: _aggregate_verdicts coverage
# ---------------------------------------------------------------------------


class TestAggregateVerdicts:
    def test_single_verdict(self) -> None:
        v = JudgeVerdict(
            a_correctness=5, a_procedure=5, a_conciseness=5,
            b_correctness=9, b_procedure=8, b_conciseness=7,
            winner="b",
        )
        candidate = _make_candidate(original="short", mutated="improved version")
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=3)
        assert score.correctness == 0.9
        assert score.procedure_following == 0.8
        assert score.conciseness == 0.7
        assert score.confidence == pytest.approx(1 / 3, abs=1e-3)

    def test_max_score_verdict(self) -> None:
        v = JudgeVerdict(
            a_correctness=5, a_procedure=5, a_conciseness=5,
            b_correctness=10, b_procedure=10, b_conciseness=10,
            winner="b",
        )
        candidate = _make_candidate()
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        assert score.correctness == 1.0
        assert score.procedure_following == 1.0
        assert score.conciseness == 1.0

    def test_length_penalty_capped_at_one(self) -> None:
        v = JudgeVerdict(
            a_correctness=5, a_procedure=5, a_conciseness=5,
            b_correctness=9, b_procedure=8, b_conciseness=7,
            winner="b",
        )
        candidate = _make_candidate(original="x", mutated="x" * 1000)
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        assert 0.0 <= score.length_penalty <= 1.0

    def test_zero_original_length(self) -> None:
        v = JudgeVerdict(
            a_correctness=5, a_procedure=5, a_conciseness=5,
            b_correctness=9, b_procedure=8, b_conciseness=7,
            winner="b",
        )
        candidate = _make_candidate(original="", mutated="content")
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        # When original is empty, length_ratio = 1.0 (no division by zero)
        assert score.length_penalty == 0.0

    def test_overall_not_below_zero(self) -> None:
        v = JudgeVerdict(
            a_correctness=0, a_procedure=0, a_conciseness=0,
            b_correctness=0, b_procedure=0, b_conciseness=0,
            winner="tie",
        )
        candidate = _make_candidate(original="x", mutated="x" * 1000)
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        assert score.overall >= 0.0

    def test_overall_not_above_one(self) -> None:
        v = JudgeVerdict(
            a_correctness=0, a_procedure=0, a_conciseness=0,
            b_correctness=10, b_procedure=10, b_conciseness=10,
            winner="b",
        )
        candidate = _make_candidate()
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        assert score.overall <= 1.0


# ---------------------------------------------------------------------------
# Test: FitnessScore confidence edge
# ---------------------------------------------------------------------------


class TestFitnessScoreEdgeCases:
    def test_zero_total_cases(self) -> None:
        score = FitnessScore(
            correctness=0.5,
            procedure_following=0.5,
            conciseness=0.5,
            overall=0.5,
            length_penalty=0.0,
            confidence=0.0,
        )
        assert score.confidence == 0.0


# ---------------------------------------------------------------------------
# Test: JudgeVerdict fields
# ---------------------------------------------------------------------------


class TestJudgeVerdictFields:
    def test_all_fields_accessible(self) -> None:
        v = JudgeVerdict(
            a_correctness=7.5,
            a_procedure=6.5,
            a_conciseness=8.0,
            b_correctness=9.0,
            b_procedure=8.5,
            b_conciseness=7.0,
            winner="b",
        )
        assert v.a_correctness == 7.5
        assert v.b_procedure == 8.5
        assert v.winner == "b"

    def test_winner_tie(self) -> None:
        v = JudgeVerdict(
            a_correctness=7,
            a_procedure=7,
            a_conciseness=7,
            b_correctness=7,
            b_procedure=7,
            b_conciseness=7,
            winner="tie",
        )
        assert v.winner == "tie"


# ---------------------------------------------------------------------------
# Test: multiple code blocks in _parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdictMultipleCodeBlocks:
    def test_multiple_code_blocks_finds_json(self) -> None:
        text = (
            "```python\nprint('hello')\n```\n\n"
            "```json\n"
            '{"a_correctness":7,"a_procedure":6,'
            '"a_conciseness":8,"b_correctness":9,"b_procedure":8,'
            '"b_conciseness":7,"winner":"a"}'
            "\n```\n\n"
            "```\nsome text\n```"
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "a"
        assert verdict.b_correctness == 9

    def test_first_code_block_no_json_second_has(self) -> None:
        text = (
            "```\nno json here\n```\n\n"
            "```json\n"
            '{"a_correctness":5,"a_procedure":5,'
            '"a_conciseness":5,"b_correctness":5,"b_procedure":5,'
            '"b_conciseness":5,"winner":"tie"}'
            "\n```"
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "tie"

    def test_code_block_with_json_prefix_no_braces_falls_through(self) -> None:
        text = (
            "```json\nno braces\n```\n"
            '{"a_correctness":8,"a_procedure":7,'
            '"a_conciseness":6,"b_correctness":9,"b_procedure":8,'
            '"b_conciseness":7,"winner":"b"}'
        )
        verdict = FitnessEvaluator._parse_verdict(text)
        assert verdict is not None
        assert verdict.winner == "b"


# ---------------------------------------------------------------------------
# Test: score aggregation with multiple verdicts
# ---------------------------------------------------------------------------


class TestAggregateVerdictsExpanded:
    def test_multiple_verdicts_average_correctly(self) -> None:
        v1 = JudgeVerdict(
            a_correctness=5, a_procedure=5, a_conciseness=5,
            b_correctness=8, b_procedure=8, b_conciseness=8,
            winner="b",
        )
        v2 = JudgeVerdict(
            a_correctness=6, a_procedure=6, a_conciseness=6,
            b_correctness=10, b_procedure=10, b_conciseness=10,
            winner="b",
        )
        candidate = _make_candidate(original="test", mutated="test mutation")
        score = FitnessEvaluator._aggregate_verdicts([v1, v2], candidate, total_cases=4)
        assert score.correctness == 0.9  # (8+10)/20
        assert score.procedure_following == 0.9
        assert score.conciseness == 0.9
        assert score.confidence == 0.5  # 2 out of 4

    def test_winner_a_does_not_affect_b_score(self) -> None:
        v1 = JudgeVerdict(
            a_correctness=10, a_procedure=10, a_conciseness=10,
            b_correctness=2, b_procedure=2, b_conciseness=2,
            winner="a",
        )
        v2 = JudgeVerdict(
            a_correctness=3, a_procedure=3, a_conciseness=3,
            b_correctness=8, b_procedure=8, b_conciseness=8,
            winner="b",
        )
        candidate = _make_candidate()
        score = FitnessEvaluator._aggregate_verdicts([v1, v2], candidate, total_cases=2)
        assert score.correctness == 0.5  # (2+8)/20
        assert score.confidence == 1.0

    def test_tie_verdict_does_not_penalize(self) -> None:
        v = JudgeVerdict(
            a_correctness=7, a_procedure=7, a_conciseness=7,
            b_correctness=7, b_procedure=7, b_conciseness=7,
            winner="tie",
        )
        candidate = _make_candidate()
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        assert score.correctness == 0.7
        assert score.overall > 0.0

    def test_overall_with_full_length_penalty(self) -> None:
        v = JudgeVerdict(
            a_correctness=0, a_procedure=0, a_conciseness=0,
            b_correctness=10, b_procedure=10, b_conciseness=10,
            winner="b",
        )
        candidate = _make_candidate(original="x", mutated="x" * 1000)
        score = FitnessEvaluator._aggregate_verdicts([v], candidate, total_cases=1)
        assert score.overall < 1.0  # length penalty reduces it


# ---------------------------------------------------------------------------
# Test: _judge exception at call_llm level
# ---------------------------------------------------------------------------


class TestJudgeLlmException:
    @pytest.mark.asyncio
    async def test_call_llm_raises_exception(self) -> None:
        evaluator = FitnessEvaluator()
        with patch.object(evaluator, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = await evaluator._judge(
                purpose="test",
                original="original",
                mutated="mutated",
            )
            assert result is None


# ---------------------------------------------------------------------------
# Test: evaluate with judge returning partial failures
# ---------------------------------------------------------------------------


class TestEvaluateMixedResults:
    @pytest.mark.asyncio
    async def test_some_cases_return_none_from_judge(self) -> None:
        evaluator = FitnessEvaluator()
        candidate = _make_candidate()

        call_count = 0

        async def _mock_judge(purpose, original, mutated):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return None
            return JudgeVerdict(
                a_correctness=5, a_procedure=5, a_conciseness=5,
                b_correctness=9, b_procedure=8, b_conciseness=7,
                winner="b",
            )

        with patch.object(evaluator, "_judge", side_effect=_mock_judge):
            score = await evaluator.evaluate(candidate, test_cases=["T1", "T2", "T3"])
            assert score.confidence == pytest.approx(2 / 3, abs=0.01)
            assert score.overall > 0.0
