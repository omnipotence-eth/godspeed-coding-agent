"""Tests for skill auto-generation from repeated tool patterns."""

from __future__ import annotations

from godspeed.evolution.skill_gen import GeneratedSkill, SkillGenerator
from godspeed.evolution.trace_analyzer import SessionTrace, ToolCall, ToolSequence

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sequence(
    tools: tuple[str, ...] = ("file_read", "file_edit"),
    frequency: int = 5,
) -> ToolSequence:
    return ToolSequence(
        tools=tools,
        frequency=frequency,
        avg_success_rate=0.9,
        candidate_skill_name="_and_".join(tools),
    )


# ---------------------------------------------------------------------------
# Test: generate_skill_markdown
# ---------------------------------------------------------------------------


class TestGenerateSkillMarkdown:
    def test_generates_valid_markdown(self) -> None:
        gen = SkillGenerator()
        seq = _sequence()
        markdown = gen.generate_skill_markdown(seq)

        assert "---" in markdown
        assert "file_read" in markdown
        assert "file_edit" in markdown

    def test_validates_own_output(self) -> None:
        gen = SkillGenerator()
        seq = _sequence()
        markdown = gen.generate_skill_markdown(seq)
        assert gen.validate_skill(markdown) is True

    def test_custom_description(self) -> None:
        gen = SkillGenerator()
        seq = _sequence()
        markdown = gen.generate_skill_markdown(seq, description="Custom desc")
        assert "Custom desc" in markdown


# ---------------------------------------------------------------------------
# Test: generate (full GeneratedSkill)
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_generated_skill(self) -> None:
        gen = SkillGenerator()
        seq = _sequence()
        skill = gen.generate(seq)

        assert isinstance(skill, GeneratedSkill)
        assert skill.source_tools == ("file_read", "file_edit")
        assert skill.frequency == 5
        assert len(skill.content) > 0

    def test_name_is_kebab_case(self) -> None:
        gen = SkillGenerator()
        seq = _sequence(tools=("file_read", "file_edit"))
        skill = gen.generate(seq)
        assert "-" in skill.name
        assert "_" not in skill.name


# ---------------------------------------------------------------------------
# Test: validate_skill
# ---------------------------------------------------------------------------


class TestValidateSkill:
    def test_valid_skill(self) -> None:
        gen = SkillGenerator()
        valid = "---\nname: test\ndescription: A test\ntrigger: test\n---\n\n1. Do something"
        assert gen.validate_skill(valid) is True

    def test_missing_frontmatter(self) -> None:
        gen = SkillGenerator()
        assert gen.validate_skill("No frontmatter here") is False

    def test_missing_required_field(self) -> None:
        gen = SkillGenerator()
        invalid = "---\nname: test\n---\n\nContent"
        assert gen.validate_skill(invalid) is False

    def test_empty_body(self) -> None:
        gen = SkillGenerator()
        invalid = "---\nname: test\ndescription: d\ntrigger: t\n---\n"
        assert gen.validate_skill(invalid) is False


# ---------------------------------------------------------------------------
# Test: detect_patterns (delegates to TraceAnalyzer)
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    def test_delegates_to_analyzer(self) -> None:
        gen = SkillGenerator()
        # Create sessions with repeated patterns
        tc_read = ToolCall(
            tool_name="file_read",
            arguments={},
            output_length=100,
            is_error=False,
            latency_ms=50.0,
            outcome="success",
        )
        tc_edit = ToolCall(
            tool_name="file_edit",
            arguments={},
            output_length=100,
            is_error=False,
            latency_ms=50.0,
            outcome="success",
        )

        sessions = [
            SessionTrace(
                session_id=f"s{i}",
                tool_calls=(tc_read, tc_edit),
                errors=(),
                permission_denials=(),
                permission_grants=(),
                total_latency_ms=100.0,
                model="",
            )
            for i in range(5)
        ]

        patterns = gen.detect_patterns(sessions, min_frequency=3)
        assert len(patterns) > 0
        pair = next((p for p in patterns if p.tools == ("file_read", "file_edit")), None)
        assert pair is not None


# ---------------------------------------------------------------------------
# Test: _make_name helper
# ---------------------------------------------------------------------------


class TestMakeName:
    def test_deduplicates_consecutive(self) -> None:
        name = SkillGenerator._make_name(("file_read", "file_read", "file_edit"))
        assert name == "file-read-and-file-edit"

    def test_caps_length(self) -> None:
        long_tools = tuple(f"tool_{i}" for i in range(20))
        name = SkillGenerator._make_name(long_tools)
        assert len(name) <= 50
