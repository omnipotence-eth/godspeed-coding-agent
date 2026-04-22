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


class TestPostEditSyntaxGate:
    """Syntax gate: edits that break a previously-parseable file must be rejected.

    Protects against the "multi-line replace drops indentation" failure mode
    observed in the daily-use benchmark (benchmark T4: user_id -> account_id
    rename de-indented function body and produced a SyntaxError).
    """

    @pytest.mark.asyncio
    async def test_py_indent_strip_rejected(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        """Fuzzy-match edit that drops indentation on a .py file must be rejected."""
        f = tool_context.cwd / "api.py"
        f.write_text(
            'def handle(req):\n    uid = req["user_id"]\n    return uid\n',
            encoding="utf-8",
        )
        # Agent provides new_string without the leading 4-space indent.
        # Fuzzy match will still find it (whitespace drift), but writing
        # would produce an IndentationError. The gate must reject.
        result = await tool.execute(
            {
                "file_path": "api.py",
                "old_string": '    uid = req["user_id"]\n    return uid',
                "new_string": 'uid = req["account_id"]\nreturn uid',
            },
            tool_context,
        )
        assert result.is_error, f"expected reject, got: {result.output}"
        assert "syntax" in (result.error or "").lower()
        # File content must be unchanged
        assert f.read_text(encoding="utf-8").count("user_id") == 1

    @pytest.mark.asyncio
    async def test_py_correctly_indented_replacement_allowed(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        """A properly-indented replacement passes the gate."""
        f = tool_context.cwd / "api.py"
        f.write_text(
            'def handle(req):\n    uid = req["user_id"]\n    return uid\n',
            encoding="utf-8",
        )
        result = await tool.execute(
            {
                "file_path": "api.py",
                "old_string": '    uid = req["user_id"]\n    return uid',
                "new_string": '    uid = req["account_id"]\n    return uid',
            },
            tool_context,
        )
        assert not result.is_error, f"expected success, got: {result.error}"
        assert 'req["account_id"]' in f.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_py_already_broken_allows_edit(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        """If file was already broken, gate doesn't block an edit (agent may be fixing it)."""
        f = tool_context.cwd / "broken.py"
        f.write_text("def foo(:\n    return 1\n", encoding="utf-8")  # pre-existing syntax error
        result = await tool.execute(
            {
                "file_path": "broken.py",
                "old_string": "def foo(:",
                "new_string": "def foo():",
            },
            tool_context,
        )
        assert not result.is_error
        # After the fix, file should parse
        import ast

        ast.parse(f.read_text(encoding="utf-8"))

    @pytest.mark.asyncio
    async def test_non_py_file_skips_gate(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        """Non-.py files are not gated — an edit that would break Python syntax
        in a .txt file should still go through."""
        f = tool_context.cwd / "notes.txt"
        f.write_text("def foo():\n    return 1\n", encoding="utf-8")
        result = await tool.execute(
            {
                "file_path": "notes.txt",
                "old_string": "return 1",
                "new_string": "return 1 +",  # would be Python-invalid
            },
            tool_context,
        )
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_json_broken_rejected(
        self, tool: FileEditTool, tool_context: ToolContext
    ) -> None:
        f = tool_context.cwd / "config.json"
        f.write_text('{"a": 1, "b": 2}\n', encoding="utf-8")
        # Drop the closing quote on a key — breaks JSON
        result = await tool.execute(
            {
                "file_path": "config.json",
                "old_string": '"b": 2',
                "new_string": '"b: 2',
            },
            tool_context,
        )
        assert result.is_error
        assert "JSON" in (result.error or "")
        # Content unchanged
        assert f.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'

    @pytest.mark.asyncio
    async def test_json_valid_allowed(self, tool: FileEditTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "config.json"
        f.write_text('{"a": 1}\n', encoding="utf-8")
        result = await tool.execute(
            {
                "file_path": "config.json",
                "old_string": '{"a": 1}',
                "new_string": '{"a": 2}',
            },
            tool_context,
        )
        assert not result.is_error
        assert '{"a": 2}' in f.read_text(encoding="utf-8")
