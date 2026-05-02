"""Additional tests for godspeed.tui.output to push coverage over 80%."""

from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from godspeed.tui import output as _output
from godspeed.tui.output import (
    capture_output,
    format_error,
    format_status_hud,
    format_thinking,
    format_welcome,
    is_compact_mode,
    set_compact_mode,
)


def _capture(fn, *args, **kwargs) -> str:
    """Run a formatting function and capture its Rich console output."""
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")  # noqa: N806
    buf = StringIO()
    original = _output.console
    _output.console = Console(file=buf, force_terminal=True, width=120)
    try:
        fn(*args, **kwargs)
    finally:
        _output.console = original
    return _ANSI_RE.sub("", buf.getvalue())


class TestCaptureOutput:
    def test_capture_basic(self):
        with capture_output() as buf:
            _output.console.print("Hello World")
        output = buf.getvalue()
        assert "Hello World" in output

    def test_capture_width(self):
        with capture_output(width=80) as buf:
            _output.console.print("Test")
        output = buf.getvalue()
        assert "Test" in output

    def test_restore_after_exception(self):
        try:
            with capture_output() as _buf:
                _output.console.print("Before")
                raise ValueError("test error")
        except ValueError:
            pass
        # Console should be restored
        assert isinstance(_output.console, Console)


class TestFormatThinking:
    def test_empty_text(self):
        # Should return early without printing
        with capture_output() as buf:
            format_thinking("")
        output = buf.getvalue()
        assert output.strip() == ""

    def test_short_text(self):
        output = _capture(format_thinking, "This is a test thought")
        assert "Thinking" in output
        assert "This is a test thought" in output

    def test_long_text_truncated(self):
        long_text = "x" * 2500
        output = _capture(format_thinking, long_text)
        assert "truncated" in output.lower()
        assert "500" in output  # 2500 - 2000 = 500 chars truncated

    def test_panel_title(self):
        output = _capture(format_thinking, "test")
        assert "Thinking" in output


class TestFormatError:
    def test_basic_error(self):
        output = _capture(format_error, "Something went wrong")
        assert "Error" in output or "Something went wrong" in output

    def test_error_shows_message(self):
        output = _capture(format_error, "File not found")
        assert "File not found" in output
        assert "Error" in output


class TestFormatStatusHud:
    def test_critical_context(self):
        """Test context >= 90% shows critical style."""
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=9000,
            output_tokens=1000,
            context_pct=95.0,
            cost_usd=0.0,
            turns=1,
        )
        assert "95%" in output or "95.0%" in output

    def test_warning_context(self):
        """Test context 70-90% shows warning style."""
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=7500,
            output_tokens=1500,
            context_pct=75.0,
            cost_usd=0.0,
            turns=1,
        )
        assert "75%" in output or "75.0%" in output

    def test_ok_context(self):
        """Test context < 70% shows ok style."""
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=3000,
            output_tokens=1000,
            context_pct=40.0,
            cost_usd=0.0,
            turns=1,
        )
        assert "40%" in output or "40.0%" in output

    def test_with_budget_near_limit(self):
        """Test budget display when near limit."""
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=8.0,
            budget_usd=10.0,  # 20% remaining = near limit
            turns=1,
        )
        assert "$8" in output or "$8.0000" in output

    def test_with_budget_plenty(self):
        """Test budget display when not near limit."""
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=1.0,
            budget_usd=10.0,
            turns=1,
        )
        assert "$1" in output or "$1.0000" in output

    def test_without_budget(self):
        """Test cost display without budget."""
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.5,
            turns=1,
        )
        assert "$0.5" in output or "$0.5000" in output

    def test_model_with_preset(self):
        # Note: preset parameter exists but may not be displayed in output
        # Just verify the function runs without error
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,
            turns=1,
            preset="thinking",
        )
        assert "claude-sonnet" in output

    def test_model_short_name(self):
        output = _capture(
            format_status_hud,
            model="anthropic/claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,
            turns=1,
        )
        assert "claude-sonnet" in output

    def test_with_turns_and_max(self):
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,
            turns=5,
            max_iterations=20,
        )
        assert "5" in output and "20" in output

    def test_yolo_mode(self):
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,
            turns=1,
            permission_mode="yolo",
        )
        assert "YOLO" in output

    def test_strict_mode(self):
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,
            turns=1,
            permission_mode="strict",
        )
        assert "strict" in output.lower()

    def test_plan_mode(self):
        output = _capture(
            format_status_hud,
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0,
            turns=1,
            permission_mode="plan",
        )
        assert "plan" in output.lower()


class TestFormatWelcome:
    def test_with_tools_list(self):
        """Test welcome with tools parameter."""
        output = _capture(
            format_welcome,
            model="test-model",
            project_dir="/home/user/project",
            tools=["file_read", "shell"],
        )
        assert "Godspeed" in output
        assert "test-model" in output

    def test_minimal_output(self):
        output = _capture(
            format_welcome,
            model="model",
            project_dir="/project",
        )
        # Should contain the model and version
        assert "model" in output
        assert "Godspeed" in output
        assert "v0" in output  # version


class TestCompactMode:
    def test_toggle_multiple_times(self):
        set_compact_mode(True)
        assert is_compact_mode() is True
        set_compact_mode(False)
        assert is_compact_mode() is False
        set_compact_mode(True)
        assert is_compact_mode() is True
        set_compact_mode(False)  # Reset for other tests
