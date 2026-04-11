"""Shared test fixtures for Godspeed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from godspeed.config import GodspeedSettings
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult


class MockTool(Tool):
    """A minimal tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        risk_level: RiskLevel = RiskLevel.READ_ONLY,
        result: ToolResult | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._risk_level = risk_level
        self._result = result or ToolResult.success("mock output")
        self.last_arguments: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk_level

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"},
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.last_arguments = arguments
        return self._result


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with .godspeed/."""
    godspeed_dir = tmp_path / ".godspeed"
    godspeed_dir.mkdir()
    return tmp_path


@pytest.fixture
def tool_context(tmp_project: Path) -> ToolContext:
    """Create a ToolContext for testing."""
    return ToolContext(cwd=tmp_project, session_id="test-session-001")


@pytest.fixture
def settings(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> GodspeedSettings:
    """Create GodspeedSettings with isolated config (no real config files)."""
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_project / ".godspeed-global")
    monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_project / ".godspeed")
    return GodspeedSettings(project_dir=tmp_project)
