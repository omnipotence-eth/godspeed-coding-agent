"""Tests for the evolution engine — GEPA-style LLM-guided mutations."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from godspeed.evolution.mutator import (
    EvolutionEngine,
    MutationCandidate,
    SkillCandidate,
)
from godspeed.evolution.trace_analyzer import (
    EvolutionReport,
    ToolCall,
    ToolFailurePattern,
    ToolSequence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_failure(
    tool_name: str = "bash",
    category: str = "execution_error",
    frequency: int = 5,
) -> ToolFailurePattern:
    return ToolFailurePattern(
        tool_name=tool_name,
        error_category=category,
        frequency=frequency,
        example_args=({"cmd": "ls -la"},),
        suggested_fix=f"Tool '{tool_name}' description may need better examples",
    )


def _make_report(
    error_rate: float = 0.2,
    failures: tuple[ToolFailurePattern, ...] = (),
) -> EvolutionReport:
    return EvolutionReport(
        sessions_analyzed=10,
        tool_failures=failures,
        latency_stats=(),
        permission_insights=(),
        tool_sequences=(),
        most_used_tools=(("file_read", 50), ("bash", 30)),
        error_rate=error_rate,
    )


def _make_tool_call(
    tool_name: str = "file_read",
    is_error: bool = False,
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        arguments={"path": "src/foo.py"},
        output_length=200,
        is_error=is_error,
        latency_ms=45.0,
        outcome="success" if not is_error else "error",
    )


def _make_sequence() -> ToolSequence:
    return ToolSequence(
        tools=("file_read", "file_edit"),
        frequency=8,
        avg_success_rate=0.9,
        candidate_skill_name="file_read_and_file_edit",
    )


# ---------------------------------------------------------------------------
# Test: MutationCandidate data structure
# ---------------------------------------------------------------------------


class TestMutationCandidate:
    def test_frozen(self) -> None:
        mc = MutationCandidate(
            artifact_type="tool_description",
            artifact_id="bash",
            original="old",
            mutated="new",
            mutation_rationale="fix errors",
            model_used="ollama/gemma3:12b",
        )
        with pytest.raises(AttributeError):
            mc.mutated = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        mc = MutationCandidate(
            artifact_type="prompt_section",
            artifact_id="core",
            original="a",
            mutated="b",
            mutation_rationale="r",
            model_used="m",
        )
        assert mc.artifact_type == "prompt_section"
        assert mc.artifact_id == "core"


# ---------------------------------------------------------------------------
# Test: mutate_tool_description
# ---------------------------------------------------------------------------


class TestMutateToolDescription:
    @pytest.mark.asyncio
    async def test_returns_candidates(self) -> None:
        engine = EvolutionEngine(model="ollama/test")
        failure = _make_failure()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Improved description for bash tool with examples."

            candidates = await engine.mutate_tool_description(
                tool_name="bash",
                current_desc="Run shell commands.",
                failure_patterns=[failure],
                num_candidates=2,
            )

        assert len(candidates) == 2
        assert all(isinstance(c, MutationCandidate) for c in candidates)
        assert candidates[0].artifact_type == "tool_description"
        assert candidates[0].artifact_id == "bash"
        assert candidates[0].original == "Run shell commands."
        assert candidates[0].mutated == "Improved description for bash tool with examples."

    @pytest.mark.asyncio
    async def test_no_failures_returns_empty(self) -> None:
        engine = EvolutionEngine()
        candidates = await engine.mutate_tool_description(
            tool_name="bash",
            current_desc="Run shell commands.",
            failure_patterns=[],
        )
        assert candidates == []

    @pytest.mark.asyncio
    async def test_identical_mutation_skipped(self) -> None:
        engine = EvolutionEngine()
        failure = _make_failure()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # Return identical text — should be skipped
            mock_llm.return_value = "Run shell commands."

            candidates = await engine.mutate_tool_description(
                tool_name="bash",
                current_desc="Run shell commands.",
                failure_patterns=[failure],
                num_candidates=2,
            )

        assert candidates == []

    @pytest.mark.asyncio
    async def test_llm_error_handled(self) -> None:
        engine = EvolutionEngine()
        failure = _make_failure()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM down")

            candidates = await engine.mutate_tool_description(
                tool_name="bash",
                current_desc="Run shell commands.",
                failure_patterns=[failure],
                num_candidates=2,
            )

        assert candidates == []

    @pytest.mark.asyncio
    async def test_rationale_includes_failure_count(self) -> None:
        engine = EvolutionEngine()
        failures = [_make_failure(), _make_failure(category="timeout")]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Better description."

            candidates = await engine.mutate_tool_description(
                tool_name="bash",
                current_desc="old",
                failure_patterns=failures,
                num_candidates=1,
            )

        assert "2 failure pattern" in candidates[0].mutation_rationale


# ---------------------------------------------------------------------------
# Test: mutate_prompt_section
# ---------------------------------------------------------------------------


class TestMutatePromptSection:
    @pytest.mark.asyncio
    async def test_returns_candidates(self) -> None:
        engine = EvolutionEngine()
        report = _make_report(
            error_rate=0.3,
            failures=(_make_failure(),),
        )

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Improved core prompt section."

            candidates = await engine.mutate_prompt_section(
                section_name="core",
                current_text="You are a coding agent.",
                report=report,
                num_candidates=1,
            )

        assert len(candidates) == 1
        assert candidates[0].artifact_type == "prompt_section"
        assert "30.0%" in candidates[0].mutation_rationale

    @pytest.mark.asyncio
    async def test_llm_error_handled(self) -> None:
        engine = EvolutionEngine()
        report = _make_report()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("fail")

            candidates = await engine.mutate_prompt_section(
                section_name="core",
                current_text="prompt",
                report=report,
                num_candidates=1,
            )

        assert candidates == []


# ---------------------------------------------------------------------------
# Test: mutate_compaction_prompt
# ---------------------------------------------------------------------------


class TestMutateCompactionPrompt:
    @pytest.mark.asyncio
    async def test_returns_candidates(self) -> None:
        engine = EvolutionEngine()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Better compaction prompt."

            candidates = await engine.mutate_compaction_prompt(
                current_prompt="Summarize the conversation.",
                quality_scores=[0.6, 0.7, 0.5],
                num_candidates=1,
            )

        assert len(candidates) == 1
        assert candidates[0].artifact_type == "compaction_prompt"
        assert "0.60" in candidates[0].mutation_rationale


# ---------------------------------------------------------------------------
# Test: generate_tool_examples
# ---------------------------------------------------------------------------


class TestGenerateToolExamples:
    @pytest.mark.asyncio
    async def test_generates_examples(self) -> None:
        engine = EvolutionEngine()
        traces = [_make_tool_call(), _make_tool_call()]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (
                '- Example: Read a Python file\n  Arguments: {"path": "src/main.py"}'
            )

            examples = await engine.generate_tool_examples("file_read", traces)

        assert len(examples) >= 1
        assert "Example" in examples[0]

    @pytest.mark.asyncio
    async def test_empty_traces_returns_empty(self) -> None:
        engine = EvolutionEngine()
        examples = await engine.generate_tool_examples("file_read", [])
        assert examples == []


# ---------------------------------------------------------------------------
# Test: suggest_new_skill
# ---------------------------------------------------------------------------


class TestSuggestNewSkill:
    @pytest.mark.asyncio
    async def test_generates_skill(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()
        traces = [[_make_tool_call("file_read"), _make_tool_call("file_edit")]]

        skill_yaml = """---
name: read-and-edit
description: Read then edit a file
trigger: read-edit
---

1. Read the file with file_read
2. Edit the file with file_edit"""

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = skill_yaml

            result = await engine.suggest_new_skill(sequence, traces)

        assert result is not None
        assert isinstance(result, SkillCandidate)
        assert result.name == "read-and-edit"
        assert result.trigger == "read-edit"
        assert "file_read" in result.content

    @pytest.mark.asyncio
    async def test_malformed_output_returns_none(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "No frontmatter here"

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None


# ---------------------------------------------------------------------------
# Test: model configuration
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_default_model_auto_detects(self) -> None:
        with patch("godspeed.evolution.hardware.detect_vram_mb", return_value=14000):
            engine = EvolutionEngine()
            assert engine.model == "ollama/gemma3:12b"

    def test_default_model_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware.detect_vram_mb", return_value=4000):
            engine = EvolutionEngine()
            assert engine.model == "ollama/qwen2.5:3b"

    def test_custom_model(self) -> None:
        engine = EvolutionEngine(model="anthropic/claude-sonnet-4-20250514")
        assert engine.model == "anthropic/claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Test: _format_failures helper
# ---------------------------------------------------------------------------


class TestFormatFailures:
    def test_formats_correctly(self) -> None:
        failures = [_make_failure(), _make_failure(category="timeout", frequency=3)]
        result = EvolutionEngine._format_failures(failures)
        assert "execution_error" in result
        assert "timeout" in result
        assert "5x" in result
        assert "3x" in result

    def test_empty_list(self) -> None:
        result = EvolutionEngine._format_failures([])
        assert result == ""
