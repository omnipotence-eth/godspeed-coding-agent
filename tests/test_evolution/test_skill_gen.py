"""Tests for skill auto-generation — detect repeated tool patterns and create skills."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml

from godspeed.evolution.skill_gen import GeneratedSkill, SkillGenerator
from godspeed.evolution.trace_analyzer import SessionTrace, ToolCall, ToolSequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool_sequence(
    tools: tuple[str, ...] = ("file_read", "file_edit"),
    frequency: int = 5,
    avg_success_rate: float = 0.9,
    candidate_skill_name: str = "file_read_and_file_edit",
) -> ToolSequence:
    return ToolSequence(
        tools=tools,
        frequency=frequency,
        avg_success_rate=avg_success_rate,
        candidate_skill_name=candidate_skill_name,
    )


def _make_session_trace(
    session_id: str = "sess-1",
    tool_names: tuple[str, ...] = ("file_read", "bash", "file_edit"),
) -> SessionTrace:
    tool_calls = tuple(
        ToolCall(
            tool_name=name,
            arguments={},
            output_length=100,
            is_error=False,
            latency_ms=50.0,
            outcome="success",
        )
        for name in tool_names
    )
    return SessionTrace(
        session_id=session_id,
        tool_calls=tool_calls,
        errors=(),
        permission_denials=(),
        permission_grants=(),
        total_latency_ms=150.0,
        model="test-model",
    )


# ---------------------------------------------------------------------------
# Test: detect_patterns
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    def test_delegates_to_trace_analyzer(self) -> None:
        sessions = [_make_session_trace()]
        mock_seq = _make_tool_sequence()

        generator = SkillGenerator()
        generator._analyzer = MagicMock(
            analyze_multi_tool_sequences=MagicMock(return_value=[mock_seq])
        )
        result = generator.detect_patterns(sessions, min_frequency=5)
        generator._analyzer.analyze_multi_tool_sequences.assert_called_once_with(sessions, 5)
        assert result == [mock_seq]

    def test_default_min_frequency(self) -> None:
        sessions = [_make_session_trace()]

        generator = SkillGenerator()
        mock_seq = _make_tool_sequence()
        generator._analyzer = MagicMock(
            analyze_multi_tool_sequences=MagicMock(return_value=[mock_seq])
        )
        result = generator.detect_patterns(sessions)
        generator._analyzer.analyze_multi_tool_sequences.assert_called_once_with(sessions, 3)
        assert len(result) == 1
        assert result[0] is mock_seq


# ---------------------------------------------------------------------------
# Test: generate_skill_markdown
# ---------------------------------------------------------------------------


class TestGenerateSkillMarkdown:
    def test_with_default_description(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(tools=("file_read", "file_edit"))

        result = generator.generate_skill_markdown(seq)

        assert result.startswith("---\n")
        assert "---" in result[4:]
        frontmatter_part = result.split("---")[1]
        metadata = yaml.safe_load(frontmatter_part)
        assert metadata["name"] == "file-read-and-file-edit"
        assert metadata["description"] == "Auto-generated: file read then file edit"
        assert metadata["trigger"] == "file-read-and-file-edit"
        body_part = "---".join(result.split("---")[2:]).strip()
        assert "Use `file_read` tool" in body_part
        assert "Use `file_edit` tool" in body_part

    def test_with_custom_description(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(tools=("bash", "file_read"))
        custom_desc = "Run a shell command then read the output file"

        result = generator.generate_skill_markdown(seq, description=custom_desc)

        metadata = yaml.safe_load(result.split("---")[1])
        assert metadata["description"] == custom_desc
        assert metadata["name"] == "bash-and-file-read"

    def test_with_single_tool(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(tools=("file_read",), candidate_skill_name="file_read")

        result = generator.generate_skill_markdown(seq)
        body_part = "---".join(result.split("---")[2:]).strip()
        assert body_part == "1. Use `file_read` tool"

    def test_with_three_tools(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(
            tools=("file_read", "grep", "file_edit"),
            candidate_skill_name="file_read_and_grep_and_file_edit",
        )

        result = generator.generate_skill_markdown(seq)
        body_part = "---".join(result.split("---")[2:]).strip()
        lines = body_part.split("\n")
        assert len(lines) == 3
        assert "1. Use `file_read` tool" in lines[0]
        assert "2. Use `grep` tool" in lines[1]
        assert "3. Use `file_edit` tool" in lines[2]


# ---------------------------------------------------------------------------
# Test: generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generated_skill_default_description(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(
            tools=("file_read", "file_edit"), frequency=7, avg_success_rate=0.85
        )

        result = generator.generate(seq)

        assert isinstance(result, GeneratedSkill)
        assert result.name == "file-read-and-file-edit"
        assert result.description == "Auto-generated: file read then file edit"
        assert result.trigger == "file-read-and-file-edit"
        assert result.source_tools == ("file_read", "file_edit")
        assert result.frequency == 7
        assert "---\n" in result.content
        assert "Use `file_read` tool" in result.content

    def test_generated_skill_custom_description(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(tools=("bash", "grep"))
        custom_desc = "Shell search pattern"

        result = generator.generate(seq, description=custom_desc)

        assert result.description == custom_desc
        assert result.name == "bash-and-grep"
        assert result.trigger == "bash-and-grep"
        assert result.source_tools == ("bash", "grep")

    def test_generated_skill_passes_content(self) -> None:
        generator = SkillGenerator()
        seq = _make_tool_sequence(tools=("file_read",))

        result = generator.generate(seq)

        assert result.content.startswith("---\n")
        assert "1. Use `file_read` tool" in result.content


# ---------------------------------------------------------------------------
# Test: validate_skill
# ---------------------------------------------------------------------------


class TestValidateSkill:
    def test_valid_skill(self) -> None:
        generator = SkillGenerator()
        skill_text = (
            "---\nname: test-skill\ndescription: A test\ntrigger: test\n---\n\n1. Do something\n"
        )
        assert generator.validate_skill(skill_text) is True

    def test_no_frontmatter(self) -> None:
        generator = SkillGenerator()
        assert generator.validate_skill("Just plain text without any frontmatter") is False

    def test_not_enough_parts(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\nname: test-skill\n"
        assert generator.validate_skill(skill_text) is False

    def test_invalid_yaml(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\nname: [unclosed\n---\n\nBody here.\n"
        assert generator.validate_skill(skill_text) is False

    def test_metadata_not_dict(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\n- list item\n- another\n---\n\nBody.\n"
        assert generator.validate_skill(skill_text) is False

    def test_missing_required_field_name(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\ndescription: Missing name\ntrigger: test\n---\n\nBody.\n"
        assert generator.validate_skill(skill_text) is False

    def test_missing_required_field_description(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\nname: test-skill\ntrigger: test\n---\n\nBody.\n"
        assert generator.validate_skill(skill_text) is False

    def test_missing_required_field_trigger(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\nname: test-skill\ndescription: A test\n---\n\nBody.\n"
        assert generator.validate_skill(skill_text) is False

    def test_empty_body(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\nname: test-skill\ndescription: A test\ntrigger: test\n---\n\n\n"
        assert generator.validate_skill(skill_text) is False

    def test_body_only_whitespace(self) -> None:
        generator = SkillGenerator()
        skill_text = "---\nname: test-skill\ndescription: A test\ntrigger: test\n---\n   \n"
        assert generator.validate_skill(skill_text) is False

    def test_multiple_frontmatter_blocks(self) -> None:
        generator = SkillGenerator()
        skill_text = (
            "---\n"
            "name: test-skill\n"
            "description: A test\n"
            "trigger: test\n"
            "---\n"
            "---\n"
            "more yaml\n"
            "---\n"
            "\n"
            "Body.\n"
        )
        assert generator.validate_skill(skill_text) is True

    def test_extra_fields_allowed(self) -> None:
        generator = SkillGenerator()
        skill_text = (
            "---\n"
            "name: test-skill\n"
            "description: A test\n"
            "trigger: test\n"
            "extra: allowed\n"
            "---\n"
            "\n"
            "Body.\n"
        )
        assert generator.validate_skill(skill_text) is True


# ---------------------------------------------------------------------------
# Test: _make_name
# ---------------------------------------------------------------------------


class TestMakeName:
    def test_simple_name(self) -> None:
        result = SkillGenerator._make_name(("file_read", "bash"))
        assert result == "file-read-and-bash"

    def test_deduplicates_consecutive_tools(self) -> None:
        result = SkillGenerator._make_name(("file_read", "file_read", "bash", "bash"))
        assert result == "file-read-and-bash"

    def test_non_consecutive_duplicates_preserved(self) -> None:
        result = SkillGenerator._make_name(("file_read", "bash", "file_read"))
        assert result == "file-read-and-bash-and-file-read"

    def test_caps_at_50_chars(self) -> None:
        tools = tuple(f"very_long_tool_name_number_{i}" for i in range(10))
        result = SkillGenerator._make_name(tools)
        assert len(result) <= 50

    def test_replaces_underscores_with_hyphens(self) -> None:
        result = SkillGenerator._make_name(("my_tool", "other_tool"))
        assert result == "my-tool-and-other-tool"

    def test_sanitizes_non_alphanumeric(self) -> None:
        result = SkillGenerator._make_name(("tool@name!", "bash#script"))
        assert result == "toolname-and-bashscript"

    def test_lowercases(self) -> None:
        result = SkillGenerator._make_name(("FileRead", "BASH"))
        assert result == "fileread-and-bash"

    def test_single_tool(self) -> None:
        result = SkillGenerator._make_name(("file_read",))
        assert result == "file-read"

    def test_empty_tools(self) -> None:
        result = SkillGenerator._make_name(tuple())
        assert result == ""


# ---------------------------------------------------------------------------
# Test: _make_description
# ---------------------------------------------------------------------------


class TestMakeDescription:
    def test_basic_description(self) -> None:
        result = SkillGenerator._make_description(("file_read", "file_edit"))
        assert result == "Auto-generated: file read then file edit"

    def test_deduplicates_tool_names(self) -> None:
        result = SkillGenerator._make_description(("file_read", "file_read", "bash"))
        assert result == "Auto-generated: file read then bash"

    def test_single_tool(self) -> None:
        result = SkillGenerator._make_description(("file_read",))
        assert result == "Auto-generated: file read"

    def test_replaces_underscores_with_spaces(self) -> None:
        result = SkillGenerator._make_description(("my_tool",))
        assert result == "Auto-generated: my tool"

    def test_preserves_order_of_first_occurrence(self) -> None:
        result = SkillGenerator._make_description(("bash", "file_read", "bash"))
        assert result == "Auto-generated: bash then file read"


# ---------------------------------------------------------------------------
# Test: GeneratedSkill dataclass
# ---------------------------------------------------------------------------


class TestGeneratedSkill:
    def test_frozen(self) -> None:
        skill = GeneratedSkill(
            name="test",
            description="desc",
            trigger="test",
            content="body",
            source_tools=("tool_a",),
            frequency=3,
        )
        with pytest.raises(AttributeError):
            skill.name = "changed"  # type: ignore[misc]

    def test_all_fields(self) -> None:
        skill = GeneratedSkill(
            name="my-skill",
            description="A generated skill",
            trigger="my-skill",
            content="---\nname: my-skill\n---\n\nBody.\n",
            source_tools=("file_read", "file_edit"),
            frequency=12,
        )
        assert skill.name == "my-skill"
        assert skill.description == "A generated skill"
        assert skill.trigger == "my-skill"
        assert skill.content == "---\nname: my-skill\n---\n\nBody.\n"
        assert skill.source_tools == ("file_read", "file_edit")
        assert skill.frequency == 12
