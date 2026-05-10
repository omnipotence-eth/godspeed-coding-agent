"""Tests for grep search tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.grep_search import MAX_MATCHES, GrepSearchTool, _search_file


@pytest.fixture
def tool() -> GrepSearchTool:
    return GrepSearchTool()


def _write_file(root: Path, name: str, content: str) -> Path:
    """Write a file under root, creating parent dirs as needed."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestSearchFile:
    """Unit tests for _search_file helper."""

    def test_no_match_in_file(self) -> None:
        import re

        path = Path("dummy")
        with patch.object(Path, "read_text", return_value="no match here\n"):
            result = _search_file(path, re.compile("xyz"), 2)
            assert result == []

    def test_unicode_decode_error_returns_empty(self) -> None:
        import re

        path = Path("dummy")
        with patch.object(
            Path, "read_text", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "reason")
        ):
            result = _search_file(path, re.compile("."), 2)
            assert result == []

    def test_permission_error_returns_empty(self) -> None:
        import re

        path = Path("dummy")
        with patch.object(Path, "read_text", side_effect=PermissionError):
            result = _search_file(path, re.compile("."), 2)
            assert result == []

    def test_multiple_blocks(self) -> None:
        import re

        path = Path("dummy")
        content = "a\nb\nmatch1\nc\nd\ne\nf\nmatch2\ng\n"
        with patch.object(Path, "read_text", return_value=content):
            result = _search_file(path, re.compile("match"), 1)
        assert len(result) >= 1

    def test_adjacent_matches_single_block(self) -> None:
        import re

        path = Path("dummy")
        content = "match1\nmatch2\nc\n"
        with patch.object(Path, "read_text", return_value=content):
            result = _search_file(path, re.compile("match"), 1)
        assert len(result) == 1  # Both matches in same block

    def test_single_line_file(self) -> None:
        import re

        path = Path("dummy")
        with patch.object(Path, "read_text", return_value="only match\n"):
            result = _search_file(path, re.compile("match"), 2)
        assert len(result) == 1


class TestGrepSearchTool:
    """Test regex content searching."""

    def test_metadata(self, tool: GrepSearchTool) -> None:
        assert tool.name == "grep_search"
        assert tool.risk_level == "read_only"
        assert "Search file contents" in tool.description

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

    @pytest.mark.asyncio
    async def test_context_lines_invalid_type(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "hello\n")
        result = await tool.execute({"pattern": "hello", "context_lines": "invalid"}, tool_context)
        assert result.is_error
        assert "context_lines must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_context_lines_float(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "hello\n")
        result = await tool.execute({"pattern": "hello", "context_lines": 2.0}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_context_lines_negative_clamped(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "hello\n")
        result = await tool.execute({"pattern": "hello", "context_lines": -5}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_path_not_exists(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"pattern": "hello", "path": "nonexistent_dir"}, tool_context)
        assert result.is_error
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_outside_project(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        result = await tool.execute({"pattern": "hello", "path": "..\\outside"}, tool_context)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_search_directory(self, tool: GrepSearchTool, tool_context: ToolContext) -> None:
        sub = tool_context.cwd / "sub"
        sub.mkdir()
        _write_file(tool_context.cwd, "sub/a.py", "hello\n")
        _write_file(tool_context.cwd, "sub/b.py", "world\n")
        result = await tool.execute({"pattern": "hello", "path": "sub"}, tool_context)
        assert not result.is_error
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_default_path_is_cwd(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "unique123\n")
        result = await tool.execute({"pattern": "unique123"}, tool_context)
        assert not result.is_error
        assert "unique123" in result.output

    @pytest.mark.asyncio
    async def test_max_matches_truncation(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        for i in range(MAX_MATCHES + 10):
            _write_file(tool_context.cwd, f"file_{i}.py", f"MATCH_{i}\n")
        result = await tool.execute({"pattern": "MATCH_", "context_lines": 0}, tool_context)
        assert not result.is_error
        assert "truncated" in result.output

    @pytest.mark.asyncio
    async def test_glob_default_all_files(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "a.py", "hello\n")
        _write_file(tool_context.cwd, "b.txt", "hello\n")
        result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error
        assert "a.py" in result.output
        assert "b.txt" in result.output

    @pytest.mark.asyncio
    async def test_safe_filter_oserror_handled(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "good.py", "hello\n")
        with patch("pathlib.Path.glob", side_effect=OSError("disk error")):
            result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_safe_filter_handles_path_errors(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        """Test _safe_filter gracefully handles OSError/PermissionError/ValueError."""
        from unittest.mock import MagicMock

        _write_file(tool_context.cwd, "good.py", "hello\n")
        bad_path = MagicMock()
        bad_path.is_file.side_effect = PermissionError("access denied")
        bad_path.__hash__.return_value = 0
        bad_path.__lt__ = lambda self, other: False
        bad_path.__gt__ = lambda self, other: False

        with patch.object(type(tool_context.cwd), "glob", return_value=[bad_path]):
            result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_glob_oserror_handled(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "a.py", "hello\n")
        with patch.object(type(tool_context.cwd), "glob", side_effect=OSError("glob failed")):
            result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error
        assert "No matches" in result.output

    # --- Additional block/context tests ---

    def test_block_split_with_gap(self) -> None:
        import re

        path = Path("dummy")
        content = "a\nMATCH1\nb\nc\nd\ne\nMATCH2\nf\n"
        with patch.object(Path, "read_text", return_value=content):
            result = _search_file(path, re.compile("MATCH"), 0)
        assert len(result) == 2

    def test_single_line_multi_block(self) -> None:
        import re

        path = Path("dummy")
        content = "M1\nx\nM2\n"
        with patch.object(Path, "read_text", return_value=content):
            result = _search_file(path, re.compile("M[12]"), 0)
        assert len(result) == 2

    def test_three_blocks_with_gaps(self) -> None:
        import re

        path = Path("dummy")
        content = "match\n\n\n\nmatch\n\n\n\nmatch\n"
        with patch.object(Path, "read_text", return_value=content):
            result = _search_file(path, re.compile("match"), 0)
        assert len(result) == 3

    def test_context_lines_capture_neighbors(self) -> None:
        import re

        path = Path("dummy")
        content = "\n".join(f"L{i}" for i in range(1, 11))
        with patch.object(Path, "read_text", return_value=content):
            result = _search_file(path, re.compile("L5"), 2)
        assert len(result) == 1
        block = result[0]
        assert "L3" in block
        assert "L7" in block

    # --- context_lines edge cases ---

    @pytest.mark.asyncio
    async def test_context_lines_zero(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "before\nmatch\nafter\n")
        result = await tool.execute({"pattern": "match", "context_lines": 0}, tool_context)
        assert not result.is_error
        output_lines = result.output.splitlines()
        match_lines = [l for l in output_lines if ">" in l and "match" in l]
        assert len(match_lines) >= 1

    @pytest.mark.asyncio
    async def test_context_lines_very_large(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        lines = [f"line {i}" for i in range(1, 11)]
        _write_file(tool_context.cwd, "data.txt", "\n".join(lines))
        result = await tool.execute({"pattern": "line 5", "context_lines": 999}, tool_context)
        assert not result.is_error
        assert "line 1" in result.output
        assert "line 10" in result.output

    # --- _safe_filter edge cases ---

    @pytest.mark.asyncio
    async def test_safe_filter_valueerror_on_relative_to(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "good.py", "hello\n")
        bad_path = MagicMock()
        bad_path.is_file.return_value = True
        bad_path.relative_to.side_effect = ValueError("no common prefix")
        bad_path.__hash__.return_value = 12345
        bad_path.__lt__ = lambda self, other: str(self) < str(other)
        bad_path.__gt__ = lambda self, other: str(self) > str(other)

        with patch.object(type(tool_context.cwd), "glob", return_value=[bad_path]):
            result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_safe_filter_oserror_on_relative_to(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "good.py", "hello\n")
        bad_path = MagicMock()
        bad_path.is_file.return_value = True
        bad_path.relative_to.side_effect = OSError("os error on relative_to")
        bad_path.__hash__.return_value = 54321
        bad_path.__lt__ = lambda self, other: False
        bad_path.__gt__ = lambda self, other: False

        with patch.object(type(tool_context.cwd), "glob", return_value=[bad_path]):
            result = await tool.execute({"pattern": "hello"}, tool_context)
        assert not result.is_error

    # --- MAX_MATCHES precise boundary ---

    @pytest.mark.asyncio
    async def test_max_matches_exact_boundary(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        for i in range(MAX_MATCHES):
            _write_file(tool_context.cwd, f"file_{i}.py", f"MATCH_{i}\n")
        result = await tool.execute({"pattern": "MATCH_", "context_lines": 0}, tool_context)
        assert not result.is_error
        assert "truncated" in result.output

    @pytest.mark.asyncio
    async def test_max_matches_just_below(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        for i in range(MAX_MATCHES - 1):
            _write_file(tool_context.cwd, f"file_{i}.py", f"MATCH_{i}\n")
        result = await tool.execute({"pattern": "MATCH_", "context_lines": 0}, tool_context)
        assert not result.is_error
        assert "truncated" not in result.output

    # --- relative_to ValueError in result output loop ---

    @pytest.mark.asyncio
    async def test_relative_to_fallback_on_output(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "good.py", "needle\n")

        call_count = [0]

        def _relative_to_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return tool_context.cwd / "mock_fallback"
            raise ValueError("mismatch during output formatting")

        mock_file = MagicMock()
        mock_file.relative_to = MagicMock(side_effect=_relative_to_side_effect)
        mock_file.is_file = MagicMock(return_value=True)
        mock_file.__lt__ = lambda self, other: str(self) < str(other)
        mock_file.__gt__ = lambda self, other: str(self) > str(other)

        with patch("godspeed.tools.grep_search._search_file", return_value=["needle match"]):
            with patch.object(type(tool_context.cwd), "glob", return_value=[mock_file]):
                result = await tool.execute({"pattern": "needle"}, tool_context)

        assert not result.is_error
        assert "needle" in result.output

    # --- Directory traversal with glob ---

    @pytest.mark.asyncio
    async def test_glob_multiple_dirs(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "src/module.py", "secret\n")
        _write_file(tool_context.cwd, "tests/test.py", "secret\n")
        _write_file(tool_context.cwd, "docs/readme.md", "PUBLIC\n")
        result = await tool.execute({"pattern": "secret", "glob": "**/*.py"}, tool_context)
        assert not result.is_error
        assert "module.py" in result.output
        assert "test.py" in result.output
        assert "readme.md" not in result.output

    @pytest.mark.asyncio
    async def test_glob_in_subdirectory(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        (tool_context.cwd / "nested").mkdir()
        _write_file(tool_context.cwd, "nested/deep.py", "FIXME\n")
        _write_file(tool_context.cwd, ".hidden/file.py", "FIXME\n")
        result = await tool.execute({"pattern": "FIXME", "glob": "**/*.py"}, tool_context)
        assert not result.is_error
        assert "nested" in result.output or "FIXME" in result.output

    # --- Invalid context_lines types ---

    @pytest.mark.asyncio
    async def test_context_lines_none(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "hello\n")
        result = await tool.execute({"pattern": "hello", "context_lines": None}, tool_context)
        assert result.is_error
        assert "context_lines must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_context_lines_dict(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "app.py", "hello\n")
        result = await tool.execute({"pattern": "hello", "context_lines": {"a": 1}}, tool_context)
        assert result.is_error
        assert "context_lines must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_search_single_file_with_absolute_path(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        test_file = _write_file(tool_context.cwd, "single.py", "target\n")
        abs_path = str(test_file)
        result = await tool.execute({"pattern": "target", "path": abs_path}, tool_context)
        assert not result.is_error
        assert "target" in result.output

    @pytest.mark.asyncio
    async def test_search_without_path_defaults_to_cwd(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "alone.py", "needle123\n")
        result = await tool.execute({"pattern": "needle123"}, tool_context)
        assert not result.is_error
        assert "needle123" in result.output

    @pytest.mark.asyncio
    async def test_empty_glob_returns_all(
        self, tool: GrepSearchTool, tool_context: ToolContext
    ) -> None:
        _write_file(tool_context.cwd, "a.py", "hello\n")
        _write_file(tool_context.cwd, "b.txt", "hello\n")
        _write_file(tool_context.cwd, "c.md", "hello\n")
        result = await tool.execute({"pattern": "hello", "glob": ""}, tool_context)
        assert not result.is_error
        assert "a.py" in result.output
        assert "b.txt" in result.output
        assert "c.md" in result.output
