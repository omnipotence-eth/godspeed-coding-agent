"""Tests for the TUI thinking spinner and tool tracking callbacks."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from godspeed.tui.app import _TOOL_LABELS, _on_assistant_chunk, _ThinkingSpinner


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


class TestLiveTokenHud:
    """Live HUD: counters update as chunks stream, throttled label refresh."""

    def _mock_client(self) -> MagicMock:
        client = MagicMock()
        client.total_input_tokens = 0
        client.total_cache_read_tokens = 0
        client.total_cache_creation_tokens = 0
        return client

    def test_output_chunk_increments_char_counter(self) -> None:
        spinner = _ThinkingSpinner(llm_client=self._mock_client())
        spinner.add_output_chunk("hello world")
        assert spinner._output_chars == 11

    def test_thinking_chunk_increments_thinking_counter(self) -> None:
        spinner = _ThinkingSpinner(llm_client=self._mock_client())
        spinner.add_thinking_chunk("let me think about this")
        assert spinner._thinking_chars == 23

    def test_add_chunks_noop_on_empty_text(self) -> None:
        spinner = _ThinkingSpinner(llm_client=self._mock_client())
        spinner.add_output_chunk("")
        spinner.add_thinking_chunk("")
        assert spinner._output_chars == 0
        assert spinner._thinking_chars == 0

    def test_render_label_shows_output_tokens_approx(self) -> None:
        spinner = _ThinkingSpinner(llm_client=self._mock_client())
        # 4000 chars ≈ 1000 tokens
        spinner._output_chars = 4000
        label = spinner._render_label()
        # _fmt_tokens formats 1000 as "1.00k"
        assert "↓" in label
        assert "1.00k" in label

    def test_render_label_shows_input_tokens_and_cache_pct(self) -> None:
        client = self._mock_client()
        client.total_input_tokens = 10_000
        client.total_cache_read_tokens = 9_500
        spinner = _ThinkingSpinner(llm_client=client)
        # Anchors are 0 → full deltas
        label = spinner._render_label()
        assert "↑" in label
        assert "10.0k" in label
        assert "95% cached" in label

    def test_render_label_hides_meta_when_no_activity(self) -> None:
        spinner = _ThinkingSpinner(llm_client=self._mock_client())
        label = spinner._render_label()
        assert "Thinking..." in label
        assert "↓" not in label
        assert "↑" not in label

    def test_start_anchors_counters(self) -> None:
        client = self._mock_client()
        client.total_input_tokens = 5000
        client.total_cache_read_tokens = 1000
        spinner = _ThinkingSpinner(llm_client=client)
        with patch("godspeed.tui.app.console"):
            spinner.start()
            # Simulate LLM finishing a response: totals grow by 3000
            client.total_input_tokens = 8000
            client.total_cache_read_tokens = 2500
            label = spinner._render_label()
            # Label shows delta (3000 in, 1500 cached) not 8000/2500
            assert "3.00k" in label
            assert "50% cached" in label
            spinner.stop()


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


class TestAssistantStreamCallback:
    """Streaming assistant output must not parse model text as Rich markup."""

    def test_chunk_does_not_raise_on_bracket_heavy_text(self) -> None:
        buf = StringIO()
        fake = Console(file=buf, width=120, force_terminal=True, legacy_windows=False)
        sneaky = "[/BOLD_WARNING] error-style text [bold]x[/bold] and [incomplete"
        with patch("godspeed.tui.app.console", fake):
            _on_assistant_chunk(sneaky)
        assert sneaky in buf.getvalue()
