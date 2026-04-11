"""Tests for the TUI thinking spinner and tool tracking callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from godspeed.tui.app import _TOOL_LABELS, _ThinkingSpinner


class TestThinkingSpinner:
    """Test context-aware thinking spinner."""

    def test_start_creates_status(self) -> None:
        spinner = _ThinkingSpinner()
        with patch("godspeed.tui.app.console"):
            spinner.start()
            assert spinner._started is True
            spinner.stop()

    def test_stop_clears_started(self) -> None:
        spinner = _ThinkingSpinner()
        with patch("godspeed.tui.app.console"):
            spinner.start()
            spinner.stop()
            assert spinner._started is False

    def test_stop_noop_when_not_started(self) -> None:
        spinner = _ThinkingSpinner()
        spinner.stop()  # Should not raise
        assert spinner._started is False

    def test_start_idempotent(self) -> None:
        spinner = _ThinkingSpinner()
        with patch("godspeed.tui.app.console"):
            spinner.start()
            spinner.start()  # Should not create a second spinner
            assert spinner._started is True
            spinner.stop()

    def test_update_changes_status_text(self) -> None:
        spinner = _ThinkingSpinner()
        with patch("godspeed.tui.app.console"):
            spinner.start()
            mock_status = MagicMock()
            spinner._status = mock_status
            spinner._started = True
            spinner.update("file_read", {"file_path": "src/main.py"})
            mock_status.update.assert_called_once()
            call_text = mock_status.update.call_args[0][0]
            assert "Reading" in call_text
            assert "src/main.py" in call_text
            spinner.stop()

    def test_update_truncates_long_args(self) -> None:
        spinner = _ThinkingSpinner()
        mock_status = MagicMock()
        spinner._status = mock_status
        spinner._started = True
        long_path = "a" * 100
        spinner.update("file_read", {"file_path": long_path})
        call_text = mock_status.update.call_args[0][0]
        assert "..." in call_text
        assert len(long_path) > 50  # Original was long

    def test_update_noop_when_not_started(self) -> None:
        spinner = _ThinkingSpinner()
        spinner.update("file_read", {"file_path": "x.py"})  # Should not raise

    def test_wrap_stops_spinner_before_calling_fn(self) -> None:
        spinner = _ThinkingSpinner()
        calls: list[str] = []

        def my_fn() -> None:
            calls.append("called")

        wrapped = spinner.wrap(my_fn)
        with patch("godspeed.tui.app.console"):
            spinner.start()
            wrapped()
            assert calls == ["called"]
            assert spinner._started is False

    def test_update_shell_shows_command(self) -> None:
        spinner = _ThinkingSpinner()
        mock_status = MagicMock()
        spinner._status = mock_status
        spinner._started = True
        spinner.update("shell", {"command": "pytest -v"})
        call_text = mock_status.update.call_args[0][0]
        assert "Running" in call_text
        assert "pytest -v" in call_text

    def test_update_unknown_tool_uses_name(self) -> None:
        spinner = _ThinkingSpinner()
        mock_status = MagicMock()
        spinner._status = mock_status
        spinner._started = True
        spinner.update("custom_tool", {})
        call_text = mock_status.update.call_args[0][0]
        assert "custom_tool" in call_text


class TestToolLabels:
    """Test the _TOOL_LABELS mapping."""

    def test_common_tools_have_labels(self) -> None:
        assert "file_read" in _TOOL_LABELS
        assert "file_edit" in _TOOL_LABELS
        assert "shell" in _TOOL_LABELS
        assert "grep_search" in _TOOL_LABELS

    def test_labels_are_human_readable(self) -> None:
        for label in _TOOL_LABELS.values():
            assert label[0].isupper()
            assert len(label) > 2
