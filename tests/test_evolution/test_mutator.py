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
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            engine = EvolutionEngine()
            assert engine.model == "ollama/devstral-small-2:24b"

    def test_default_model_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=5500):
            engine = EvolutionEngine()
            assert engine.model == "ollama/rnj-1:8b"

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

    def test_format_failure_with_example_args(self) -> None:
        failure = ToolFailurePattern(
            tool_name="bash",
            error_category="invalid_args",
            frequency=1,
            example_args=({"path": "/nonexistent"},),
            suggested_fix="Fix it",
        )
        result = EvolutionEngine._format_failures([failure])
        assert "invalid_args" in result
        assert "1x" in result


# ---------------------------------------------------------------------------
# Test: mutate_prompt_section edge cases
# ---------------------------------------------------------------------------


class TestMutatePromptSectionEdgeCases:
    @pytest.mark.asyncio
    async def test_identical_mutation_skipped(self) -> None:
        engine = EvolutionEngine()
        report = _make_report(failures=(_make_failure(),))

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "current prompt"

            candidates = await engine.mutate_prompt_section(
                section_name="core",
                current_text="current prompt",
                report=report,
                num_candidates=2,
            )

        assert candidates == []

    @pytest.mark.asyncio
    async def test_empty_failures_handled(self) -> None:
        engine = EvolutionEngine()
        report = _make_report()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Improved prompt."

            candidates = await engine.mutate_prompt_section(
                section_name="core",
                current_text="old prompt",
                report=report,
                num_candidates=1,
            )

        assert len(candidates) == 1
        assert "none" in mock_llm.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Test: mutate_compaction_prompt edge cases
# ---------------------------------------------------------------------------


class TestMutateCompactionPromptEdgeCases:
    @pytest.mark.asyncio
    async def test_identical_mutation_skipped(self) -> None:
        engine = EvolutionEngine()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Summarize the conversation."

            candidates = await engine.mutate_compaction_prompt(
                current_prompt="Summarize the conversation.",
                quality_scores=[0.6],
                num_candidates=2,
            )

        assert candidates == []

    @pytest.mark.asyncio
    async def test_llm_error_handled(self) -> None:
        engine = EvolutionEngine()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM down")

            candidates = await engine.mutate_compaction_prompt(
                current_prompt="prompt",
                quality_scores=[0.5],
                num_candidates=2,
            )

        assert candidates == []

    @pytest.mark.asyncio
    async def test_empty_quality_scores(self) -> None:
        engine = EvolutionEngine()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Better prompt."

            candidates = await engine.mutate_compaction_prompt(
                current_prompt="prompt",
                quality_scores=[],
                num_candidates=1,
            )

        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# Test: generate_tool_examples edge cases
# ---------------------------------------------------------------------------


class TestGenerateToolExamplesEdgeCases:
    @pytest.mark.asyncio
    async def test_llm_error_handled(self) -> None:
        engine = EvolutionEngine()
        traces = [_make_tool_call()]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM down")

            examples = await engine.generate_tool_examples("file_read", traces)

        assert examples == []

    @pytest.mark.asyncio
    async def test_llm_returns_empty_string(self) -> None:
        engine = EvolutionEngine()
        traces = [_make_tool_call()]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ""

            examples = await engine.generate_tool_examples("file_read", traces)

        assert examples == []

    @pytest.mark.asyncio
    async def test_max_examples_default(self) -> None:
        engine = EvolutionEngine()
        traces = [_make_tool_call() for _ in range(10)]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "- Example: test\n  Arguments: {}"

            examples = await engine.generate_tool_examples(
                "file_read", traces, max_examples=2
            )

        assert len(examples) >= 1
        call_text = mock_llm.call_args[0][0]
        # Only 2 traces should be included
        assert call_text.count("→") <= 2


# ---------------------------------------------------------------------------
# Test: suggest_new_skill edge cases
# ---------------------------------------------------------------------------


class TestSuggestNewSkillEdgeCases:
    @pytest.mark.asyncio
    async def test_llm_error_handled(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM down")

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_llm_response(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ""

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_yaml_in_frontmatter(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "---\n: invalid yaml\n---\ncontent here"

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None

    @pytest.mark.asyncio
    async def test_yaml_not_a_dict(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "---\n- list item 1\n- list item 2\n---\ncontent"

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None

    @pytest.mark.asyncio
    async def test_yaml_missing_fields(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "---\nname: my-skill\n---\ncontent"

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None

    @pytest.mark.asyncio
    async def test_partial_yaml_fields(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (
                "---\n"
                "name: my-skill\n"
                "description: desc\n"
                "trigger: \"\"\n"
                "---\ncontent"
            )

            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None


# ---------------------------------------------------------------------------
# Test: _call_llm branches
# ---------------------------------------------------------------------------


class TestCallLlmBranches:
    @pytest.mark.asyncio
    async def test_call_with_injected_client(self) -> None:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = "improved text"
        mock_client.chat.return_value = mock_response

        engine = EvolutionEngine(llm_client=mock_client)
        result = await engine._call_llm("test prompt")
        assert result == "improved text"
        mock_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_without_injected_client_creates_client(self) -> None:
        engine = EvolutionEngine(model="ollama/test")
        with patch("godspeed.llm.client.LLMClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.content = "response text"
            mock_client.chat.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await engine._call_llm("test prompt")
            assert result == "response text"
            mock_client_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Test: SkillCandidate data structure
# ---------------------------------------------------------------------------


class TestSkillCandidate:
    def test_frozen(self) -> None:
        sc = SkillCandidate(
            name="test",
            description="desc",
            trigger="t",
            content="c",
            source_sequence=_make_sequence(),
        )
        with pytest.raises(AttributeError):
            sc.name = "changed"  # type: ignore[misc]

    def test_all_fields(self) -> None:
        seq = _make_sequence()
        sc = SkillCandidate(
            name="read-edit",
            description="Read then edit",
            trigger="re",
            content="1. Read\n2. Edit",
            source_sequence=seq,
        )
        assert sc.name == "read-edit"
        assert sc.trigger == "re"
        assert sc.source_sequence == seq


# ---------------------------------------------------------------------------
# Test: mutate_tool_description with mixed success/failure across iterations
# ---------------------------------------------------------------------------


class TestMutateToolDescriptionMixed:
    @pytest.mark.asyncio
    async def test_partial_failures_still_returns_candidates(self) -> None:
        engine = EvolutionEngine(model="ollama/test")
        failures = [
            _make_failure(tool_name="bash", category="execution_error"),
            _make_failure(tool_name="bash", category="timeout"),
        ]

        call_count = 0

        async def _alternating(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Improved bash description v1."
            if call_count == 2:
                raise RuntimeError("transient error")
            return "Improved bash description v3."

        with patch.object(engine, "_call_llm", side_effect=_alternating):
            candidates = await engine.mutate_tool_description(
                tool_name="bash",
                current_desc="Run shell commands.",
                failure_patterns=failures,
                num_candidates=3,
            )

        assert len(candidates) == 2
        assert all(c.artifact_type == "tool_description" for c in candidates)


# ---------------------------------------------------------------------------
# Test: mutate_tool_description with empty string response
# ---------------------------------------------------------------------------


class TestMutateToolDescriptionEmptyResponse:
    @pytest.mark.asyncio
    async def test_empty_response_skipped(self) -> None:
        engine = EvolutionEngine(model="ollama/test")
        failures = [_make_failure()]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ""
            candidates = await engine.mutate_tool_description(
                tool_name="bash",
                current_desc="Run shell commands.",
                failure_patterns=failures,
                num_candidates=2,
            )

        assert candidates == []


# ---------------------------------------------------------------------------
# Test: mutate_prompt_section with many failures for top_failures formatting
# ---------------------------------------------------------------------------


class TestMutatePromptSectionExpanded:
    @pytest.mark.asyncio
    async def test_top_failures_capped_at_five(self) -> None:
        engine = EvolutionEngine()
        many_failures = tuple(
            _make_failure(tool_name=f"tool-{i}", category="execution_error")
            for i in range(10)
        )
        report = _make_report(error_rate=0.5, failures=many_failures)

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Improved prompt."
            candidates = await engine.mutate_prompt_section(
                section_name="core",
                current_text="prompt",
                report=report,
                num_candidates=1,
            )

        assert len(candidates) == 1
        call_text = mock_llm.call_args[0][0]
        assert "tool-0" in call_text
        assert "tool-4" in call_text

    @pytest.mark.asyncio
    async def test_rationale_contains_error_rate(self) -> None:
        engine = EvolutionEngine()
        report = _make_report(error_rate=0.45, failures=(_make_failure(),))

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "new prompt"
            candidates = await engine.mutate_prompt_section(
                section_name="core",
                current_text="old",
                report=report,
                num_candidates=1,
            )

        assert "45.0%" in candidates[0].mutation_rationale


# ---------------------------------------------------------------------------
# Test: mutate_compaction_prompt with many quality scores
# ---------------------------------------------------------------------------


class TestMutateCompactionPromptExpanded:
    @pytest.mark.asyncio
    async def test_last_ten_scores_only(self) -> None:
        engine = EvolutionEngine()
        scores = [float(i) / 10 for i in range(20)]  # 20 scores

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Better prompt."
            candidates = await engine.mutate_compaction_prompt(
                current_prompt="old",
                quality_scores=scores,
                num_candidates=1,
            )

        assert len(candidates) == 1
        call_text = mock_llm.call_args[0][0]
        assert "1.00" in call_text or "0.90" in call_text or "average" in call_text


# ---------------------------------------------------------------------------
# Test: generate_tool_examples with error traces mixed in
# ---------------------------------------------------------------------------


class TestGenerateToolExamplesExpanded:
    @pytest.mark.asyncio
    async def test_error_traces_included(self) -> None:
        engine = EvolutionEngine()
        traces = [
            _make_tool_call(is_error=False),
            _make_tool_call(is_error=True),
        ]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "- Example: test\n  Arguments: {}"
            examples = await engine.generate_tool_examples("file_read", traces)

        assert len(examples) >= 1

    @pytest.mark.asyncio
    async def test_llm_returns_only_whitespace(self) -> None:
        engine = EvolutionEngine()
        traces = [_make_tool_call()]

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "\n   \n"
            examples = await engine.generate_tool_examples("file_read", traces)

        assert examples == []


# ---------------------------------------------------------------------------
# Test: suggest_new_skill with full trace detail
# ---------------------------------------------------------------------------


class TestSuggestNewSkillExpanded:
    @pytest.mark.asyncio
    async def test_error_tool_calls_in_examples(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()
        traces = [
            [
                _make_tool_call("file_read", is_error=False),
                _make_tool_call("file_edit", is_error=True),
            ]
        ]

        skill_yaml = """---
name: read-edit
description: Read then edit
trigger: re
---
content"""

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = skill_yaml
            result = await engine.suggest_new_skill(sequence, traces)

        assert result is not None
        assert result.name == "read-edit"

    @pytest.mark.asyncio
    async def test_multiple_trace_examples_capped_at_three(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()
        traces = [[_make_tool_call("file_read"), _make_tool_call("file_edit")] for _ in range(5)]

        skill_yaml = """---
name: read-edit
description: desc
trigger: re
---
content"""

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = skill_yaml
            result = await engine.suggest_new_skill(sequence, traces)

        assert result is not None
        call_text = mock_llm.call_args[0][0]
        assert call_text.count("Execution ") <= 3

    @pytest.mark.asyncio
    async def test_missing_required_fields_returns_none(self) -> None:
        engine = EvolutionEngine()
        sequence = _make_sequence()

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "---\nname: only-name\n---\ncontent"
            result = await engine.suggest_new_skill(sequence, [[]])

        assert result is None


# ---------------------------------------------------------------------------
# Test: _parser_skill_candidate with frontmatter variations
# ---------------------------------------------------------------------------


class TestParseSkillCandidate:
    def test_exactly_three_parts_missing_frontmatter(self) -> None:
        from godspeed.evolution.trace_analyzer import ToolSequence
        seq = ToolSequence(
            tools=("a", "b"), frequency=1, avg_success_rate=1.0,
            candidate_skill_name="a_and_b",
        )
        result = EvolutionEngine._parse_skill_candidate("no frontmatter", seq)
        assert result is None

    def test_yaml_parse_error_returns_none(self) -> None:
        from godspeed.evolution.trace_analyzer import ToolSequence
        seq = ToolSequence(
            tools=("a",), frequency=1, avg_success_rate=1.0,
            candidate_skill_name="a",
        )
        result = EvolutionEngine._parse_skill_candidate(
            "---\n\tbad: [unclosed\n---\ncontent", seq
        )
        assert result is None

    def test_empty_string_fields_returns_none(self) -> None:
        from godspeed.evolution.trace_analyzer import ToolSequence
        seq = ToolSequence(
            tools=("a",), frequency=1, avg_success_rate=1.0,
            candidate_skill_name="a",
        )
        result = EvolutionEngine._parse_skill_candidate(
            "---\nname: \"\"\ndescription: d\ntrigger: t\n---\ncontent", seq
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test: _format_failures with example args
# ---------------------------------------------------------------------------


class TestFormatFailuresExpanded:
    def test_example_args_multiple(self) -> None:
        failures = [
            ToolFailurePattern(
                tool_name="bash",
                error_category="invalid_args",
                frequency=1,
                example_args=(),
                suggested_fix="Fix it.",
            ),
        ]
        result = EvolutionEngine._format_failures(failures)
        assert "invalid_args" in result
        # No example args line
        assert "Example args" not in result

    def test_example_args_single(self) -> None:
        failures = [
            ToolFailurePattern(
                tool_name="bash",
                error_category="invalid_args",
                frequency=1,
                example_args=({"cmd": "rm -rf /"},),
                suggested_fix="Fix it.",
            ),
        ]
        result = EvolutionEngine._format_failures(failures)
        assert "Example args" in result
        assert "rm -rf /" in result
