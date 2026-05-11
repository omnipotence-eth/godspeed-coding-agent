"""Tests for the traceback analyzer tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.traceback_analyzer import (
    TracebackAnalyzerTool,
    _extract_js_error,
    _extract_python_error,
    _format_frame,
    _format_summary,
    _parse_js_stack,
    _parse_python_traceback,
    _read_source_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx_with_dir(tmp_path: Path) -> ToolContext:
    py_file = tmp_path / "test.py"
    py_file.write_text("def foo():\n    return bar()\n\ndef bar():\n    return 1/0\n")
    return ToolContext(cwd=tmp_path, session_id="test")


@pytest.fixture
def ctx_empty(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="empty")


# ============================================================================
# Tool metadata
# ============================================================================


class TestTracebackAnalyzerMetadata:
    def test_name(self) -> None:
        tool = TracebackAnalyzerTool()
        assert tool.name == "traceback_analyzer"

    def test_risk_level(self) -> None:
        tool = TracebackAnalyzerTool()
        assert tool.risk_level.value == "read_only"

    def test_schema(self) -> None:
        tool = TracebackAnalyzerTool()
        schema = tool.get_schema()
        assert "traceback" in schema["properties"]
        assert "traceback" in schema.get("required", [])
        assert schema["properties"]["traceback"]["type"] == "string"

    def test_description(self) -> None:
        tool = TracebackAnalyzerTool()
        assert "Python" in tool.description


# ============================================================================
# Tool.execute — top-level
# ============================================================================


@pytest.mark.asyncio
async def test_parse_python_traceback(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "Traceback (most recent call last):\n"
        '  File "test.py", line 2, in foo\n'
        "    return bar()\n"
        '  File "test.py", line 5, in bar\n'
        "    return 1/0\n"
        "ZeroDivisionError: division by zero\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_with_dir)
    assert not result.is_error
    assert "ZeroDivisionError" in result.output
    assert "test.py" in result.output


@pytest.mark.asyncio
async def test_empty_traceback(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    result = await tool.execute({"traceback": ""}, ctx_with_dir)
    assert result.is_error


@pytest.mark.asyncio
async def test_missing_parameter(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    result = await tool.execute({}, ctx_with_dir)
    assert result.is_error


@pytest.mark.asyncio
async def test_whitespace_only_traceback(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    result = await tool.execute({"traceback": "   \n  \n"}, ctx_with_dir)
    assert result.is_error


@pytest.mark.asyncio
async def test_non_string_traceback(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    result = await tool.execute({"traceback": 42}, ctx_with_dir)
    assert result.is_error


# ============================================================================
# Python traceback parsing
# ============================================================================


class TestParsePythonTraceback:
    def test_basic_frames(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            '  File "app.py", line 10, in main',
            "    result = compute(x)",
            '  File "app.py", line 5, in compute',
            "    return x / 0",
            "ZeroDivisionError: division by zero",
        ]
        frames = _parse_python_traceback(lines)
        assert len(frames) == 2
        assert frames[0]["file"] == "app.py"
        assert frames[0]["line"] == 10
        assert frames[0]["function"] == "main"
        assert frames[0]["code_context"] == "result = compute(x)"
        assert frames[1]["line"] == 5
        assert frames[1]["function"] == "compute"
        assert frames[1]["code_context"] == "return x / 0"

    def test_frame_without_code_context(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            '  File "app.py", line 10, in main',
            '  File "app.py", line 5, in compute',
            "  File ...",
            "ValueError: bad value",
        ]
        frames = _parse_python_traceback(lines)
        assert len(frames) == 2
        assert frames[0]["code_context"] == ""

    def test_code_context_truncation(self) -> None:
        long_line = "    " + ("x" * 250)
        lines = [
            "Traceback (most recent call last):",
            '  File "app.py", line 1, in f',
            long_line,
            "RuntimeError: oh no",
        ]
        frames = _parse_python_traceback(lines)
        assert len(frames) == 1
        assert len(frames[0]["code_context"]) == 200

    def test_non_matching_lines_ignored(self) -> None:
        lines = [
            "Some preamble text",
            "Traceback (most recent call last):",
            '  File "app.py", line 3, in f',
            "    pass",
            "",
            "Error: stuff",
        ]
        frames = _parse_python_traceback(lines)
        assert len(frames) == 1

    def test_empty_input(self) -> None:
        assert _parse_python_traceback([]) == []

    def test_frame_as_last_line(self) -> None:
        """Last frame with no following lines — code_context stays empty."""
        lines = [
            '  File "app.py", line 42, in crash',
        ]
        frames = _parse_python_traceback(lines)
        assert len(frames) == 1
        assert frames[0]["code_context"] == ""


# ============================================================================
# Python error extraction
# ============================================================================


class TestExtractPythonError:
    def test_standard_error(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            '  File "a.py", line 1, in f',
            "    x",
            "NameError: name 'x' is not defined",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "NameError"
        assert emsg == "name 'x' is not defined"

    def test_syntax_error(self) -> None:
        lines = [
            '  File "a.py", line 1',
            "    def foo(",
            "            ^",
            "SyntaxError: '(' was never closed",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "SyntaxError"

    def test_chained_exception(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            '  File "a.py", line 2, in f',
            "    x = 1/0",
            "ZeroDivisionError: division by zero",
            "",
            "During handling of the above exception, another exception occurred:",
            "",
            "Traceback (most recent call last):",
            '  File "a.py", line 4, in f',
            "    raise ValueError('bad') from e",
            "ValueError: bad",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "ValueError"

    def test_fallback_simple_match(self) -> None:
        lines = [
            '  File "a.py", line 1, in f',
            "CustomError: something broke",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "CustomError"
        assert emsg == "something broke"

    def test_fallback_no_colon(self) -> None:
        lines = ["SomeError"]
        etype, emsg = _extract_python_error(lines)
        assert etype == "SomeError"
        assert emsg == ""

    def test_unknown_error(self) -> None:
        lines = ["just some text", "no error pattern here"]
        etype, emsg = _extract_python_error(lines)
        assert etype == "Unknown"
        assert emsg == ""

    def test_skip_traceback_and_empty_lines(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            "",
            "ValueError: oops",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "ValueError"

    def test_import_error_with_dotted_name(self) -> None:
        lines = [
            "Traceback (most recent call last):",
            "ImportError: cannot import name 'foo' from 'bar'",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "ImportError"

    def test_frame_lines_before_error(self) -> None:
        """When only skip-able lines exist, return Unknown."""
        lines = [
            "Traceback (most recent call last):",
            '  File "app.py", line 1, in f',
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "Unknown"
        assert emsg == ""

    def test_simple_match_non_error_type(self) -> None:
        """Simple regex fallback when not a standard Python error name."""
        lines = [
            "MyPrefix: something happened",
        ]
        etype, emsg = _extract_python_error(lines)
        assert etype == "MyPrefix"
        assert emsg == "something happened"


# ============================================================================
# JS/TS stack parsing
# ============================================================================


class TestParseJsStack:
    def test_basic_js_stack(self) -> None:
        lines = [
            "TypeError: Cannot read property 'x' of undefined",
            "    at Object.<anonymous> (/app/main.js:10:15)",
            "    at Module._compile (internal/modules/cjs/loader.js:999:30)",
        ]
        frames = _parse_js_stack(lines)
        assert len(frames) == 2
        assert frames[0]["file"] == "/app/main.js"
        assert frames[0]["line"] == 10
        assert frames[0]["function"] == "Object.<anonymous>"

    def test_js_anonymous_function(self) -> None:
        lines = [
            "    at /app/lib.js:5:12",
        ]
        frames = _parse_js_stack(lines)
        assert len(frames) == 1
        assert frames[0]["function"] == "<anonymous>"
        assert frames[0]["file"] == "/app/lib.js"

    def test_js_no_matching_lines(self) -> None:
        lines = [
            "Some error without stack",
            "Just a message",
        ]
        frames = _parse_js_stack(lines)
        assert frames == []

    def test_empty_js_lines(self) -> None:
        assert _parse_js_stack([]) == []


# ============================================================================
# JS error extraction
# ============================================================================


class TestExtractJsError:
    def test_standard_js_error(self) -> None:
        lines = [
            "TypeError: Cannot read property 'x' of undefined",
            "    at foo (app.js:10:5)",
        ]
        etype, emsg = _extract_js_error(lines)
        assert etype == "TypeError"
        assert emsg == "Cannot read property 'x' of undefined"

    def test_js_error_no_suffix(self) -> None:
        lines = ["RangeError: invalid array length"]
        etype, emsg = _extract_js_error(lines)
        assert etype == "RangeError"

    def test_generic_name_with_colon(self) -> None:
        lines = ["MyAppError: something went wrong"]
        etype, emsg = _extract_js_error(lines)
        assert etype == "MyAppError"
        assert emsg == "something went wrong"

    def test_just_message(self) -> None:
        lines = ["Uncaught exception"]
        etype, emsg = _extract_js_error(lines)
        assert etype == "Error"
        assert emsg == "Uncaught exception"

    def test_empty_lines(self) -> None:
        etype, emsg = _extract_js_error([])
        assert etype == "Error"
        assert emsg == ""

    def test_generic_name_not_error_suffix(self) -> None:
        """Identifier ending in something other than Error/Exception."""
        lines = ["MyTag: some message"]
        etype, emsg = _extract_js_error(lines)
        assert etype == "MyTag"
        assert emsg == "some message"


# ============================================================================
# JS traceback execution path
# ============================================================================


@pytest.mark.asyncio
async def test_parse_js_traceback(ctx_empty: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "TypeError: Cannot read property 'x' of undefined\n"
        "    at processResult (/app/handler.js:42:18)\n"
        "    at /app/index.js:15:3\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_empty)
    assert not result.is_error
    assert "TypeError" in result.output
    assert "/app/handler.js" in result.output


@pytest.mark.asyncio
async def test_js_no_frames_returns_error(ctx_empty: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = "Something went wrong\nNo stack trace available\n"
    result = await tool.execute({"traceback": tb}, ctx_empty)
    assert result.is_error
    assert "Could not parse any stack frames" in result.error


# ============================================================================
# _format_summary
# ============================================================================


class TestFormatSummary:
    def test_known_error(self) -> None:
        out = _format_summary("ZeroDivisionError", "division by zero")
        assert "TRACEBACK ANALYSIS" in out
        assert "ZeroDivisionError" in out
        assert "Division by zero" in out

    def test_syntax_error_with_hint(self) -> None:
        out = _format_summary("SyntaxError", "unexpected EOF")
        assert "SYNTAX ERROR" in out
        assert "python parser" in out.lower()

    def test_unknown_error_type(self) -> None:
        out = _format_summary("WeirdError", "something odd")
        assert "WeirdError" in out
        assert "Message:" in out

    def test_no_message(self) -> None:
        out = _format_summary("ValueError", "")
        assert "ValueError" in out
        assert "Message:" not in out

    def test_empty_error_type_with_message(self) -> None:
        out = _format_summary("", "generic failure")
        assert "generic failure" in out
        # no emoji + type line, just message line


# ============================================================================
# _format_frame
# ============================================================================


class TestFormatFrame:
    def test_root_cause_frame(self) -> None:
        frame = {"file": "app.py", "line": 42, "function": "crash", "code_context": "bang()"}
        out = _format_frame(1, 3, frame, "src context here", is_root_cause=True)
        assert "ROOT CAUSE" in out
        assert "root cause — last frame before crash" in out
        assert "app.py:42" in out
        assert "Code: bang()" in out
        assert "src context here" in out

    def test_non_root_frame(self) -> None:
        frame = {"file": "lib.py", "line": 10, "function": "helper", "code_context": ""}
        out = _format_frame(0, 3, frame, "", is_root_cause=False)
        assert "Frame 1/3" in out
        assert "helper" in out
        assert "ROOT CAUSE" not in out

    def test_frame_without_code_context(self) -> None:
        frame = {"file": "x.py", "line": 1, "function": "f"}
        out = _format_frame(0, 1, frame, "", is_root_cause=False)
        assert "Code:" not in out

    def test_frame_with_source_context(self) -> None:
        frame = {"file": "x.py", "line": 1, "function": "f", "code_context": "pass"}
        out = _format_frame(0, 1, frame, "   >>>    1 | pass", is_root_cause=False)
        assert "Source context" in out
        assert ">>>" in out


# ============================================================================
# _read_source_context
# ============================================================================


class TestReadSourceContext:
    def test_reads_context_around_line(self, ctx_with_dir: ToolContext) -> None:
        out = _read_source_context("test.py", 2, ctx_with_dir)
        assert "def foo" in out
        assert "def bar" in out

    def test_file_outside_project(self, ctx_empty: ToolContext) -> None:
        out = _read_source_context("/etc/passwd", 1, ctx_empty)
        assert "outside project" in out

    def test_file_not_found(self, ctx_empty: ToolContext) -> None:
        out = _read_source_context("nonexistent.py", 1, ctx_empty)
        assert "file not found" in out

    def test_not_a_regular_file(self, ctx_empty: ToolContext) -> None:
        subdir = ctx_empty.cwd / "subdir"
        subdir.mkdir()
        out = _read_source_context("subdir", 1, ctx_empty)
        assert "not a regular file" in out

    def test_binary_file(self, ctx_empty: ToolContext) -> None:
        bin_file = ctx_empty.cwd / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x80\xff")
        out = _read_source_context("data.bin", 1, ctx_empty)
        assert "binary or non-UTF-8" in out

    def test_unreadable_file(self, ctx_empty: ToolContext) -> None:
        bad_file = ctx_empty.cwd / "bad.txt"
        bad_file.write_text("content")
        out = _read_source_context("bad.txt", 1, ctx_empty)
        assert "def foo" in out or "content" in out

    def test_line_zero_clamped(self, ctx_with_dir: ToolContext) -> None:
        out = _read_source_context("test.py", 0, ctx_with_dir)
        assert "def foo" in out

    def test_line_after_end(self, ctx_with_dir: ToolContext) -> None:
        out = _read_source_context("test.py", 100, ctx_with_dir)
        assert "lines above" in out.lower()

    def test_ellipsis_below(self, ctx_empty: ToolContext) -> None:
        long_file = ctx_empty.cwd / "long.py"
        lines = [f"line_{i}\n" for i in range(50)]
        long_file.write_text("".join(lines))
        out = _read_source_context("long.py", 1, ctx_empty)
        assert "more lines" in out.lower() or "..." in out

    def test_ellipsis_above(self, ctx_empty: ToolContext) -> None:
        long_file = ctx_empty.cwd / "long2.py"
        lines = [f"line_{i}\n" for i in range(50)]
        long_file.write_text("".join(lines))
        out = _read_source_context("long2.py", 49, ctx_empty)
        assert "lines above" in out.lower()

    def test_os_error_during_read(self, ctx_empty: ToolContext) -> None:
        good_file = ctx_empty.cwd / "exists.txt"
        good_file.write_text("hello")

        original_read_text = Path.read_text

        def _failing_read_text(self, *args, **kwargs):
            if self.name == "exists.txt":
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", _failing_read_text):
            out = _read_source_context("exists.txt", 1, ctx_empty)
            assert "cannot read file" in out


# ============================================================================
# Integration: full analysis output
# ============================================================================


@pytest.mark.asyncio
async def test_full_python_analysis(ctx_with_dir: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "Traceback (most recent call last):\n"
        '  File "test.py", line 2, in foo\n'
        "    return bar()\n"
        '  File "test.py", line 5, in bar\n'
        "    return 1/0\n"
        "ZeroDivisionError: division by zero\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_with_dir)
    assert not result.is_error
    output = result.output
    assert "TRACEBACK ANALYSIS" in output
    assert "ZeroDivisionError" in output
    assert "ROOT CAUSE" in output
    assert "division by zero" in output


@pytest.mark.asyncio
async def test_syntax_error_analysis(ctx_empty: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 1, in <module>\n'
        "SyntaxError: '(' was never closed\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_empty)
    assert not result.is_error
    assert "SYNTAX ERROR" in result.output


@pytest.mark.asyncio
async def test_chained_exception_analysis(ctx_empty: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "Traceback (most recent call last):\n"
        '  File "outer.py", line 3, in outer\n'
        "    inner()\n"
        '  File "outer.py", line 7, in inner\n'
        "    1/0\n"
        "ZeroDivisionError: division by zero\n"
        "\n"
        "The above exception was the direct cause of the following:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "outer.py", line 10, in <module>\n'
        "    outer()\n"
        "RuntimeError: something bad\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_empty)
    assert not result.is_error
    assert "RuntimeError" in result.output


@pytest.mark.asyncio
async def test_import_error_analysis(ctx_empty: ToolContext) -> None:
    tool = TracebackAnalyzerTool()
    tb = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 1, in <module>\n'
        "    import non_existent_module\n"
        "ModuleNotFoundError: No module named 'non_existent_module'\n"
    )
    result = await tool.execute({"traceback": tb}, ctx_empty)
    assert not result.is_error
    assert "ModuleNotFoundError" in result.output
