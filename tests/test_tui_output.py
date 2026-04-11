"""Tests for Rich TUI output formatting — especially permission prompts."""

from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from godspeed.tui.output import (
    format_assistant_text,
    format_error,
    format_permission_denied,
    format_permission_prompt,
    format_tool_call,
    format_tool_result,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capture(fn, *args, **kwargs) -> str:
    """Run a formatting function and capture its Rich console output (ANSI stripped)."""
    import godspeed.tui.output as mod

    buf = StringIO()
    original = mod.console
    mod.console = Console(file=buf, force_terminal=True, width=120)
    try:
        fn(*args, **kwargs)
    finally:
        mod.console = original
    return _ANSI_RE.sub("", buf.getvalue())


class TestFormatPermissionPrompt:
    """Test enhanced permission prompts with contextual detail."""

    def test_file_edit_shows_diff(self) -> None:
        args = {
            "file_path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('goodbye')",
        }
        output = _capture(format_permission_prompt, "file_edit", "ASK", arguments=args)
        assert "file_edit" in output
        assert "src/main.py" in output
        assert "-print('hello')" in output
        assert "+print('goodbye')" in output

    def test_file_write_shows_preview(self) -> None:
        content = "\n".join(f"line {i}" for i in range(20))
        args = {"file_path": "output.py", "content": content}
        output = _capture(format_permission_prompt, "file_write", "ASK", arguments=args)
        assert "file_write" in output
        assert "output.py" in output
        # Syntax rendering may add ANSI codes around tokens, so check bare digits
        assert "0" in output
        assert "9" in output
        assert "more lines" in output

    def test_shell_shows_command(self) -> None:
        args = {"command": "rm -rf /tmp/build"}
        output = _capture(format_permission_prompt, "shell", "ASK", arguments=args)
        assert "shell" in output
        assert "rm -rf /tmp/build" in output

    def test_file_read_shows_path(self) -> None:
        args = {"file_path": "/etc/passwd"}
        output = _capture(format_permission_prompt, "file_read", "ASK", arguments=args)
        assert "file_read" in output
        assert "/etc/passwd" in output

    def test_grep_shows_pattern(self) -> None:
        args = {"pattern": "TODO", "path": "src/"}
        output = _capture(format_permission_prompt, "grep_search", "ASK", arguments=args)
        assert "grep_search" in output
        assert "TODO" in output

    def test_no_args_still_renders(self) -> None:
        output = _capture(format_permission_prompt, "unknown_tool", "Need permission")
        assert "unknown_tool" in output
        assert "Need permission" in output

    def test_empty_args_still_renders(self) -> None:
        output = _capture(format_permission_prompt, "file_edit", "ASK", arguments={})
        assert "file_edit" in output
        assert "Allow this tool call?" in output

    def test_file_edit_long_diff_truncated(self) -> None:
        old = "\n".join(f"old line {i}" for i in range(30))
        new = "\n".join(f"new line {i}" for i in range(30))
        args = {"file_path": "big.py", "old_string": old, "new_string": new}
        output = _capture(format_permission_prompt, "file_edit", "ASK", arguments=args)
        assert "more lines" in output


class TestFormatToolCall:
    """Test tool call display."""

    def test_basic_tool_call(self) -> None:
        output = _capture(format_tool_call, "shell", {"command": "ls -la"})
        assert "shell" in output
        assert "ls -la" in output

    def test_non_serializable_args(self) -> None:
        """Should not crash on non-JSON-serializable args."""
        output = _capture(format_tool_call, "test", {"key": object()})
        assert "test" in output


class TestFormatToolResult:
    """Test tool result display."""

    def test_success_result(self) -> None:
        output = _capture(format_tool_result, "shell", "file1.py\nfile2.py")
        assert "shell" in output
        assert "file1.py" in output

    def test_error_result(self) -> None:
        output = _capture(format_tool_result, "shell", "command not found", is_error=True)
        assert "error" in output
        assert "command not found" in output

    def test_long_result_truncated(self) -> None:
        long_text = "x" * 3000
        output = _capture(format_tool_result, "shell", long_text)
        assert "truncated" in output


class TestMiscFormatters:
    """Test other output formatters."""

    def test_format_error(self) -> None:
        output = _capture(format_error, "Something broke")
        assert "Something broke" in output

    def test_format_permission_denied(self) -> None:
        output = _capture(format_permission_denied, "shell", "blocked by policy")
        assert "shell" in output
        assert "blocked" in output

    def test_format_assistant_text_empty(self) -> None:
        output = _capture(format_assistant_text, "   ")
        assert output.strip() == ""

    def test_format_assistant_text_markdown(self) -> None:
        output = _capture(format_assistant_text, "**bold text**")
        assert "bold" in output
