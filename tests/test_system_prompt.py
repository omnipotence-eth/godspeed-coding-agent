"""Tests for system prompt assembly — asserts load-bearing sections are present."""

from __future__ import annotations

from pathlib import Path

from godspeed.agent.system_prompt import build_system_prompt
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult


class _StubTool(Tool):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "A stub tool for testing."

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult.success("ok")


def test_prompt_includes_core_workflow_and_quality() -> None:
    prompt = build_system_prompt(tools=[_StubTool()])

    # Identity + boundary
    assert "security-first coding agent" in prompt
    assert "definition" not in prompt.lower() or "done" in prompt.lower()
    assert 'What "done" means' in prompt

    # Workflows section with concrete recipes
    assert "Core workflows" in prompt
    assert "Edit an existing file" in prompt
    assert "Fix a failing test" in prompt
    assert "Multi-file change" in prompt

    # Error-recovery guidance is explicit (the load-bearing addition
    # over the old prompt — retrying failing tools blindly was a
    # top failure mode in the daily-use benchmark).
    assert "When a tool call errors" in prompt
    assert "Permission denied" in prompt
    assert "pattern not unique" in prompt

    # Code-quality defaults
    assert "Code-quality defaults" in prompt
    assert "Type hints on public functions" in prompt
    assert "Parameterized queries" in prompt

    # Anti-patterns (new in v3.5)
    assert "Anti-patterns" in prompt
    assert "Retrying a failing tool" in prompt

    # Communication style
    assert "Communication style" in prompt

    # Tools get listed
    assert "Available tools" in prompt
    assert "stub" in prompt


def test_plan_mode_appended_when_flag_set() -> None:
    prompt = build_system_prompt(tools=[_StubTool()], plan_mode=True)
    assert "Plan Mode ACTIVE" in prompt
    # Other sections still present in plan mode.
    assert "Code-quality defaults" in prompt


def test_project_instructions_still_render(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools=[_StubTool()],
        project_instructions="ALWAYS use uv, not pip.",
        cwd=tmp_path,
    )
    assert "Project instructions" in prompt
    assert "ALWAYS use uv" in prompt
    assert str(tmp_path) in prompt


def test_repo_map_summary_injected_when_provided(tmp_path: Path) -> None:
    # New in v3.5 — repo-map auto-inclusion for large projects. The
    # caller computes the summary (via context.repo_summary) and
    # passes it in; build_system_prompt just places it.
    prompt = build_system_prompt(
        tools=[_StubTool()],
        repo_map_summary="src/\n  foo.py: fn_a, fn_b\n  bar.py: ClassX",
        cwd=tmp_path,
    )
    assert "Repository map" in prompt
    assert "fn_a, fn_b" in prompt
    assert "ClassX" in prompt


def test_repo_map_summary_omitted_by_default() -> None:
    prompt = build_system_prompt(tools=[_StubTool()])
    assert "Repository map" not in prompt
