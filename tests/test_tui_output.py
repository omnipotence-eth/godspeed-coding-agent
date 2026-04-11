"""Tests for Rich TUI output formatting — Midnight Gold design refresh."""

from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from godspeed.tui.output import (
    format_assistant_text,
    format_error,
    format_permission_denied,
    format_permission_prompt,
    format_session_summary,
    format_tool_call,
    format_tool_result,
    format_welcome,
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
        assert "Allow?" in output

    def test_file_edit_long_diff_truncated(self) -> None:
        old = "\n".join(f"old line {i}" for i in range(30))
        new = "\n".join(f"new line {i}" for i in range(30))
        args = {"file_path": "big.py", "old_string": old, "new_string": new}
        output = _capture(format_permission_prompt, "file_edit", "ASK", arguments=args)
        assert "more lines" in output

    def test_permission_uses_dot_separator(self) -> None:
        output = _capture(format_permission_prompt, "shell", "ASK", arguments={"command": "ls"})
        assert "\u00b7" in output  # · dot separator


class TestFormatToolCall:
    """Test tool call display — compact and lightweight."""

    def test_shell_shows_dollar_prefix(self) -> None:
        output = _capture(format_tool_call, "shell", {"command": "ls -la"})
        assert "shell" in output
        assert "ls -la" in output

    def test_file_read_compact_inline(self) -> None:
        output = _capture(format_tool_call, "file_read", {"file_path": "src/main.py"})
        assert "file_read" in output
        assert "src/main.py" in output

    def test_grep_shows_pattern(self) -> None:
        output = _capture(format_tool_call, "grep_search", {"pattern": "TODO", "path": "src/"})
        assert "grep_search" in output
        assert "TODO" in output

    def test_file_edit_shows_path(self) -> None:
        output = _capture(
            format_tool_call,
            "file_edit",
            {"file_path": "main.py", "old_string": "a", "new_string": "b"},
        )
        assert "file_edit" in output
        assert "main.py" in output

    def test_file_write_shows_line_count(self) -> None:
        output = _capture(
            format_tool_call,
            "file_write",
            {"file_path": "out.py", "content": "line1\nline2\nline3"},
        )
        assert "file_write" in output
        assert "3 lines" in output

    def test_non_serializable_args(self) -> None:
        """Should not crash on non-JSON-serializable args."""
        output = _capture(format_tool_call, "test", {"key": object()})
        assert "test" in output

    def test_marker_present(self) -> None:
        output = _capture(format_tool_call, "file_read", {"file_path": "x.py"})
        assert "\u25b8" in output  # ▸ marker

    def test_git_tool_shows_action(self) -> None:
        output = _capture(format_tool_call, "git", {"action": "status"})
        assert "git" in output
        assert "status" in output


class TestFormatToolResult:
    """Test tool result display — success compact, errors expanded."""

    def test_success_result_short(self) -> None:
        output = _capture(format_tool_result, "shell", "file1.py\nfile2.py")
        assert "\u2713" in output  # ✓ marker
        assert "file1.py" in output

    def test_error_result(self) -> None:
        output = _capture(format_tool_result, "shell", "command not found", is_error=True)
        assert "\u2717" in output  # ✗ marker
        assert "command not found" in output

    def test_long_success_shows_line_count(self) -> None:
        long_text = "\n".join(f"line {i}" for i in range(50))
        output = _capture(format_tool_result, "shell", long_text)
        assert "50 lines" in output

    def test_empty_result(self) -> None:
        output = _capture(format_tool_result, "shell", "")
        assert "\u2713" in output

    def test_long_error_truncated(self) -> None:
        long_text = "\n".join(f"error line {i}" for i in range(30))
        output = _capture(format_tool_result, "shell", long_text, is_error=True)
        assert "more lines" in output


class TestFormatWelcome:
    """Test welcome banner — clean and minimal."""

    def test_shows_model_and_project(self) -> None:
        output = _capture(format_welcome, "claude-sonnet", "/home/user/project")
        assert "claude-sonnet" in output
        assert "/home/user/project" in output
        assert "Godspeed" in output

    def test_shows_audit_status(self) -> None:
        output = _capture(format_welcome, "model", "/home/user", audit_enabled=True)
        assert "enabled" in output

    def test_shows_help_hint(self) -> None:
        output = _capture(format_welcome, "model", "/home/user")
        assert "/help" in output

    def test_no_tools_list_by_default(self) -> None:
        """Welcome should NOT dump the full tools list."""
        output = _capture(
            format_welcome,
            "model",
            "/home/user",
            tools=["file_read", "file_write", "shell"],
        )
        # Tools should not be listed in the welcome banner
        assert "file_read" not in output


class TestFormatSessionSummary:
    """Test session summary on quit."""

    def test_shows_duration(self) -> None:
        output = _capture(format_session_summary, 754.0, 38000, 7000)
        assert "12m" in output
        assert "34s" in output

    def test_shows_tokens(self) -> None:
        output = _capture(format_session_summary, 60.0, 1000, 500)
        assert "1,500" in output

    def test_shows_cost(self) -> None:
        output = _capture(format_session_summary, 60.0, 1000, 500, cost=0.1847)
        assert "$0.1847" in output

    def test_shows_free_for_zero_cost(self) -> None:
        output = _capture(format_session_summary, 60.0, 1000, 500, cost=0.0)
        assert "free" in output

    def test_shows_tool_summary(self) -> None:
        output = _capture(
            format_session_summary,
            60.0,
            1000,
            500,
            tool_calls=23,
            tool_errors=3,
            tool_denied=2,
        )
        assert "23 calls" in output

    def test_branded_signoff(self) -> None:
        output = _capture(format_session_summary, 60.0, 0, 0)
        assert "Godspeed" in output


class TestMiscFormatters:
    """Test other output formatters."""

    def test_format_error(self) -> None:
        output = _capture(format_error, "Something broke")
        assert "Something broke" in output
        assert "\u2717" in output  # ✗ marker

    def test_format_permission_denied(self) -> None:
        output = _capture(format_permission_denied, "shell", "blocked by policy")
        assert "shell" in output
        assert "Blocked" in output

    def test_format_assistant_text_empty(self) -> None:
        output = _capture(format_assistant_text, "   ")
        assert output.strip() == ""

    def test_format_assistant_text_markdown(self) -> None:
        output = _capture(format_assistant_text, "**bold text**")
        assert "bold" in output
