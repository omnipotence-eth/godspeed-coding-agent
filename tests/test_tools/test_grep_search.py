"""Tests for grep search tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.grep_search import GrepSearchTool


@pytest.fixture
def tool() -> GrepSearchTool:
    return GrepSearchTool()


def _write_file(root: Path, name: str, content: str) -> Path:
    """Write a file under root, creating parent dirs as needed."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestGrepSearchTool:
    """Test regex content searching."""

    def test_metadata(self, tool: GrepSearchTool) -> None:
        assert tool.name == "grep_search"
        assert tool.risk_level == "read_only"

    def test_schema_has_required_pattern(self, tool: GrepSearchTool) -> None:
        schema = tool.get_schema()
        assert "pattern" in schema["properties"]
        assert schema["required"] == ["pattern"]

    @pytest.mark.asyncio
    async def test_simple_match(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        _write_file(tool_context.cwd, "app.py", "def hello():\n    return 'world'\n")
        result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error
        assert "hello" in result.output
        assert "app.py" in result.output

    @pytest.mark.asyncio
    async def test_regex_pattern(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        _write_file(tool_context.cwd, "main.py", "x = 42\ny = 100\nz = 7\n")
        result = await tool.execute({"pattern": r"\d{3}"}, tool_context)
        assert not result.is_error
        assert "100" in result.output
        # 42 and 7 are not 3-digit numbers
        assert "> " in result.output  # match indicator

    @pytest.mark.asyncio
    async def test_context_lines(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        lines = [f"line {i}" for i in range(1, 11)]
        _write_file(tool_context.cwd, "data.txt", "\n".join(lines))
        result = await tool.execute({"pattern": "line 5", "context_lines": 1}, tool_context)
        assert not result.is_error
        assert "line 4" in result.output
        assert "line 5" in result.output
        assert "line 6" in result.output

    @pytest.mark.asyncio
    async def test_no_matches(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        _write_file(tool_context.cwd, "app.py", "def hello():\n    pass\n")
        result = await tool.execute({"pattern": "zzzznotfound"}, tool_context)
        assert not result.is_error
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_invalid_regex(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"pattern": "[invalid"}, tool_context)
        assert result.is_error
        assert "Invalid regex" in result.error

    @pytest.mark.asyncio
    async def test_glob_filter(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        _write_file(tool_context.cwd, "app.py", "TODO: fix this\n")
        _write_file(tool_context.cwd, "readme.md", "TODO: update docs\n")
        result = await tool.execute({"pattern": "TODO", "glob": "*.py"}, tool_context)
        assert not result.is_error
        assert "app.py" in result.output
        assert "readme.md" not in result.output

    @pytest.mark.asyncio
    async def test_excludes_venv(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        _write_file(tool_context.cwd, "app.py", "import os\n")
        _write_file(tool_context.cwd, ".venv/lib/site.py", "import os\n")
        result = await tool.execute({"pattern": "import os"}, tool_context)
        assert not result.is_error
        assert "app.py" in result.output
        assert ".venv" not in result.output

    @pytest.mark.asyncio
    async def test_search_specific_file(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "a.py", "target line\n")
        _write_file(tool_context.cwd, "b.py", "target line\n")
        result = await tool.execute({"pattern": "target", "path": "a.py"}, tool_context)
        assert not result.is_error
        assert "target" in result.output

    @pytest.mark.asyncio
    async def test_empty_pattern(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"pattern": ""}, tool_context)
        assert result.is_error
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_match_indicator(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        """Matching lines are marked with '>', context lines with ' '."""
        _write_file(
            tool_context.cwd,
            "test.py",
            "line a\nmatch here\nline c\n",
        )
        result = await tool.execute({"pattern": "match here", "context_lines": 1}, tool_context)
        assert not result.is_error
        # The match line should have '>' prefix
        for line in result.output.splitlines():
            if "match here" in line:
                assert line.strip().startswith(">")
                break
        else:
            pytest.fail("Match line with '>' indicator not found")
