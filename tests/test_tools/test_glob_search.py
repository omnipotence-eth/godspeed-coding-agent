"""Tests for glob search tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.glob_search import GlobSearchTool


@pytest.fixture
def tool() -> GlobSearchTool:
    return GlobSearchTool()


def _create_files(root: Path, names: list[str]) -> None:
    """Create empty files under root, creating parent dirs as needed."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")


class TestGlobSearchTool:
    """Test glob pattern file searching."""

    def test_metadata(self, tool: GlobSearchTool) -> None:
        assert tool.name == "glob_search"
        assert tool.risk_level == "read_only"

    def test_schema_has_required_pattern(self, tool: GlobSearchTool) -> None:
        schema = tool.get_schema()
        assert "pattern" in schema["properties"]
        assert schema["required"] == ["pattern"]

    @pytest.mark.asyncio
    async def test_find_python_files(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        _create_files(tool_context.cwd, ["main.py", "utils.py", "readme.md"])
        result = await tool.execute({"pattern": "*.py"}, tool_context)
        assert not result.is_error
        assert "main.py" in result.output
        assert "utils.py" in result.output
        assert "readme.md" not in result.output

    @pytest.mark.asyncio
    async def test_recursive_glob(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        _create_files(tool_context.cwd, ["a.py", "src/b.py", "src/core/c.py"])
        result = await tool.execute({"pattern": "**/*.py"}, tool_context)
        assert not result.is_error
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.py" in result.output

    @pytest.mark.asyncio
    async def test_excludes_venv(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        _create_files(tool_context.cwd, ["app.py", ".venv/lib/site.py"])
        result = await tool.execute({"pattern": "**/*.py"}, tool_context)
        assert not result.is_error
        assert "app.py" in result.output
        assert "site.py" not in result.output

    @pytest.mark.asyncio
    async def test_excludes_node_modules(
        self, tool: GlobSearchTool, tool_context: ToolContext
    ) -> None:
        _create_files(tool_context.cwd, ["index.js", "node_modules/pkg/index.js"])
        result = await tool.execute({"pattern": "**/*.js"}, tool_context)
        assert not result.is_error
        assert "index.js" in result.output
        assert "node_modules" not in result.output

    @pytest.mark.asyncio
    async def test_excludes_pycache(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        _create_files(tool_context.cwd, ["app.py", "__pycache__/app.cpython-312.pyc"])
        result = await tool.execute({"pattern": "**/*"}, tool_context)
        assert not result.is_error
        assert "app.py" in result.output
        assert "__pycache__" not in result.output

    @pytest.mark.asyncio
    async def test_no_matches(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        _create_files(tool_context.cwd, ["readme.md"])
        result = await tool.execute({"pattern": "*.py"}, tool_context)
        assert not result.is_error
        assert "No files found" in result.output

    @pytest.mark.asyncio
    async def test_custom_search_path(
        self, tool: GlobSearchTool, tool_context: ToolContext
    ) -> None:
        _create_files(tool_context.cwd, ["src/a.py", "tests/b.py"])
        result = await tool.execute({"pattern": "*.py", "path": "src"}, tool_context)
        assert not result.is_error
        assert "a.py" in result.output
        assert "b.py" not in result.output

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"pattern": "*.py", "path": "nope"}, tool_context)
        assert result.is_error
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_pattern(self, tool: GlobSearchTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"pattern": ""}, tool_context)
        assert result.is_error
        assert "required" in result.error.lower()
