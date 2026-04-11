"""Tests for file_edit tool."""

from __future__ import annotations

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.file_edit import FUZZY_THRESHOLD, FileEditTool, _fuzzy_find


@pytest.fixture
def tool() -> FileEditTool:
    return FileEditTool()


class TestFileEditTool:
    """Test file editing with search/replace."""

    def test_metadata(self, tool: FileEditTool) -> None:
        assert tool.name == "file_edit"
        assert tool.risk_level == "low"

    @pytest.mark.asyncio
    async def test_exact_replace(self, tool: FileEditTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("def hello():\n    return 1\n")
        result = await tool.execute(
            {
                "file_path": "test.py",
                "old_string": "return 1",
                "new_string": "return 2",
            },
            tool_context,
        )
        assert not result.is_error
        assert "1 occurrence" in result.output
        assert f.read_text() == "def hello():\n    return 2\n"

    @pytest.mark.asyncio
    async def test_ambiguous_match_rejected(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("x = 1\ny = 1\n")
        result = await tool.execute(
            {"file_path": "test.py", "old_string": "1", "new_string": "2"},
            tool_context,
        )
        assert result.is_error
        assert "2 locations" in result.error

    @pytest.mark.asyncio
    async def test_replace_all(self, tool: FileEditTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("x = 1\ny = 1\n")
        result = await tool.execute(
            {
                "file_path": "test.py",
                "old_string": "1",
                "new_string": "2",
                "replace_all": True,
            },
            tool_context,
        )
        assert not result.is_error
        assert "2 occurrence" in result.output
        assert f.read_text() == "x = 2\ny = 2\n"

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool: FileEditTool, tool_context: ToolContext) -> None:
        result = await tool.execute(
            {
                "file_path": "missing.py",
                "old_string": "a",
                "new_string": "b",
            },
            tool_context,
        )
        assert result.is_error
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_match(self, tool: FileEditTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("hello world\n")
        result = await tool.execute(
            {
                "file_path": "test.py",
                "old_string": "completely different text that is not in the file at all",
                "new_string": "new",
            },
            tool_context,
        )
        assert result.is_error
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_same_string_rejected(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("hello\n")
        result = await tool.execute(
            {"file_path": "test.py", "old_string": "hello", "new_string": "hello"},
            tool_context,
        )
        assert result.is_error
        assert "different" in result.error.lower()


class TestFuzzyFind:
    """Test fuzzy matching for edit fallback."""

    def test_exact_match(self) -> None:
        content = "line 1\nline 2\nline 3\n"
        result = _fuzzy_find(content, "line 2")
        assert result is not None
        _, _, ratio = result
        assert ratio >= 0.99

    def test_whitespace_drift(self) -> None:
        content = "def foo():\n    return 1\n"
        # Slight whitespace difference (3 spaces instead of 4)
        search = "def foo():\n   return 1"
        result = _fuzzy_find(content, search)
        assert result is not None
        _, _, ratio = result
        assert ratio >= FUZZY_THRESHOLD

    def test_no_match(self) -> None:
        content = "completely unrelated content\n"
        search = "something entirely different here"
        result = _fuzzy_find(content, search)
        assert result is None

    def test_empty_content(self) -> None:
        assert _fuzzy_find("", "search") is None

    def test_empty_search(self) -> None:
        assert _fuzzy_find("content", "") is None


class TestConfidenceReporting:
    """Test that edit results include match confidence metadata."""

    @pytest.mark.asyncio
    async def test_exact_match_reports_confidence_1(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("x = 1\n")
        result = await tool.execute(
            {"file_path": "test.py", "old_string": "x = 1", "new_string": "x = 2"},
            tool_context,
        )
        assert not result.is_error
        assert "[match=exact confidence=1.00" in result.output

    @pytest.mark.asyncio
    async def test_replace_all_reports_count(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("a = 1\nb = 1\nc = 1\n")
        result = await tool.execute(
            {
                "file_path": "test.py",
                "old_string": "1",
                "new_string": "2",
                "replace_all": True,
            },
            tool_context,
        )
        assert not result.is_error
        assert "replacements=3" in result.output
        assert "match=exact" in result.output

    @pytest.mark.asyncio
    async def test_fuzzy_match_reports_ratio_and_line(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "test.py"
        f.write_text("def foo():\n    return 1\n")
        # Slight whitespace difference triggers fuzzy match
        result = await tool.execute(
            {
                "file_path": "test.py",
                "old_string": "def foo():\n   return 1",
                "new_string": "def foo():\n    return 2",
            },
            tool_context,
        )
        assert not result.is_error
        assert "match=fuzzy" in result.output
        assert "confidence=" in result.output
        assert "line=" in result.output
