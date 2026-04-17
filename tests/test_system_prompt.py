"""Tests for system prompt assembly — quality defaults + existing sections."""

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


def test_prompt_includes_core_and_workflow_and_quality() -> None:
    prompt = build_system_prompt(tools=[_StubTool()])

    # Existing sections still rendered
    assert "security-first coding agent" in prompt
    assert "Common Workflows" in prompt
    assert "Available Tools" in prompt
    assert "stub" in prompt

    # New quality defaults (v2.5.1)
    assert "Code Quality Defaults" in prompt
    assert "Type hints on public functions" in prompt
    assert "failing test first" in prompt
    assert "Default to no comments" in prompt
    assert "parameterized queries" in prompt
    assert "premature abstraction" in prompt


def test_plan_mode_appended_when_flag_set() -> None:
    prompt = build_system_prompt(tools=[_StubTool()], plan_mode=True)
    assert "PLAN MODE" in prompt
    # Quality defaults still present in plan mode.
    assert "Code Quality Defaults" in prompt


def test_project_instructions_still_render(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools=[_StubTool()],
        project_instructions="ALWAYS use uv, not pip.",
        cwd=tmp_path,
    )
    assert "Project Instructions" in prompt
    assert "ALWAYS use uv" in prompt
    assert str(tmp_path) in prompt
