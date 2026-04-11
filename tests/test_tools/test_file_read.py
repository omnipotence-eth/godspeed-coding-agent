"""Tests for file_read tool."""

from __future__ import annotations

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.file_read import FileReadTool


@pytest.fixture
def tool() -> FileReadTool:
    return FileReadTool()


class TestFileReadTool:
    """Test file reading."""

    def test_metadata(self, tool: FileReadTool) -> None:
        assert tool.name == "file_read"
        assert tool.risk_level == "read_only"

    @pytest.mark.asyncio
    async def test_read_file(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")
        result = await tool.execute({"file_path": "test.py"}, tool_context)
        assert not result.is_error
        assert "1\tline 1" in result.output
        assert "2\tline 2" in result.output
        assert "3\tline 3" in result.output

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"file_path": "missing.py"}, tool_context)
        assert result.is_error
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines))
        result = await tool.execute({"file_path": "test.py", "offset": 5}, tool_context)
        assert not result.is_error
        assert "5\tline 5" in result.output
        assert "1\tline 1" not in result.output

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "test.py"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines))
        result = await tool.execute({"file_path": "test.py", "limit": 3}, tool_context)
        assert not result.is_error
        assert "3\tline 3" in result.output
        assert "more lines" in result.output

    @pytest.mark.asyncio
    async def test_directory_error(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        subdir = tool_context.cwd / "subdir"
        subdir.mkdir()
        result = await tool.execute({"file_path": "subdir"}, tool_context)
        assert result.is_error
        assert "Not a file" in result.error

    @pytest.mark.asyncio
    async def test_empty_file(self, tool: FileReadTool, tool_context: ToolContext) -> None:
        test_file = tool_context.cwd / "empty.py"
        test_file.write_text("")
        result = await tool.execute({"file_path": "empty.py"}, tool_context)
        assert result.is_error
        assert "No content" in result.error
