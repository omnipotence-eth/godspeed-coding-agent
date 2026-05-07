"""Tests for src/godspeed/tui/app.py — TUI application and interactive components."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from godspeed.security.permissions import ALLOW, ASK, DENY, PermissionDecision
from godspeed.tui.app import (
    TUIApp,
    _InteractiveDiffReviewer,
    _InteractivePermissionProxy,
    _on_assistant_chunk,
    _on_assistant_text,
)


class TestInteractivePermissionProxy:
    """Tests for the interactive permission prompt handler."""

    def make_proxy(self, decision: str = ASK) -> _InteractivePermissionProxy:
        engine = MagicMock()
        engine.evaluate.return_value = PermissionDecision(decision, "test reason")
        engine.allow_rules = []
        engine.deny_rules = []
        engine.ask_rules = []
        engine.session_grants = []
        engine._tool_risk_levels = {}
        return _InteractivePermissionProxy(engine)

    def test_passes_through_allow(self) -> None:
        proxy = self.make_proxy(ALLOW)
        tool_call = MagicMock()
        tool_call.tool_name = "file_read"
        tool_call.arguments = {"file_path": "/test.txt"}
        result = proxy.evaluate(tool_call)
        assert result == ALLOW

    def test_passes_through_deny(self) -> None:
        proxy = self.make_proxy(DENY)
        tool_call = MagicMock()
        tool_call.tool_name = "shell"
        tool_call.arguments = {"command": "rm -rf /"}
        result = proxy.evaluate(tool_call)
        assert result == DENY

    @patch("godspeed.tui.app._output.console.input")
    def test_ask_user_denies(self, mock_input: MagicMock) -> None:
        mock_input.return_value = "n"
        proxy = self.make_proxy(ASK)
        tool_call = MagicMock()
        tool_call.tool_name = "file_write"
        tool_call.arguments = {"file_path": "/test.txt"}
        tool_call.format_for_permission.return_value = "FileWrite(/test.txt)"
        result = proxy.evaluate(tool_call)
        assert result == "deny"


class TestInteractiveDiffReviewer:
    """Tests for the TUI diff-review gate."""

    @pytest.mark.asyncio
    async def test_accepts_diff(self) -> None:
        reviewer = _InteractiveDiffReviewer()
        with patch("godspeed.tui.app._output.console.input", return_value="y"):
            result = await reviewer.review(
                tool_name="file_edit",
                path="/test.py",
                before="old content",
                after="new content",
            )
            assert result == "accept"

    @pytest.mark.asyncio
    async def test_rejects_diff(self) -> None:
        reviewer = _InteractiveDiffReviewer()
        with patch("godspeed.tui.app._output.console.input", return_value="n"):
            result = await reviewer.review(
                tool_name="file_edit",
                path="/test.py",
                before="old content",
                after="new content",
            )
            assert result == "reject"

    @pytest.mark.asyncio
    async def test_always_accepts_after_always(self) -> None:
        reviewer = _InteractiveDiffReviewer()
        with patch("godspeed.tui.app._output.console.input", return_value="a"):
            result = await reviewer.review(
                tool_name="file_edit",
                path="/test.py",
                before="old content",
                after="new content",
            )
            assert result == "accept"
            # Second call should auto-accept
            result2 = await reviewer.review(
                tool_name="file_write",
                path="/other.py",
                before="a",
                after="b",
            )
            assert result2 == "accept"

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_rejects(self) -> None:
        reviewer = _InteractiveDiffReviewer()
        with patch("godspeed.tui.app._output.console.input", side_effect=KeyboardInterrupt):
            result = await reviewer.review(
                tool_name="file_edit",
                path="/test.py",
                before="old",
                after="new",
            )
            assert result == "reject"


def test_on_assistant_chunk_does_not_raise() -> None:
    with patch("godspeed.tui.app._output.console.print") as mock_print:
        _on_assistant_chunk("hello")
        mock_print.assert_called_once_with("hello", end="")


def test_on_assistant_text_does_not_raise() -> None:
    with patch("godspeed.tui.app._output.console.print") as mock_print:
        _on_assistant_text("hello")
        mock_print.assert_called_once()


class TestTUIAppPermissionMode:
    """Tests for TUIApp._get_permission_mode()."""

    def make_app(self, mode: str = "normal") -> TUIApp:
        llm_client = MagicMock()
        tool_registry = MagicMock()
        tool_context = MagicMock()
        conversation = MagicMock()
        permission_engine = MagicMock()
        permission_engine.deny_rules = []
        permission_engine.ask_rules = [MagicMock()]
        permission_engine.session_grants = []
        permission_engine.plan_mode = False
        permission_engine._mode = mode

        return TUIApp(
            llm_client=llm_client,
            tool_registry=tool_registry,
            tool_context=tool_context,
            conversation=conversation,
            permission_engine=permission_engine,
            audit_trail=None,
            session_id="test-session",
        )

    def test_normal_mode(self) -> None:
        app = self.make_app("normal")
        mode = app._get_permission_mode()
        assert mode == "normal"

    def test_yolo_mode(self) -> None:
        app = self.make_app("normal")
        app._permission_engine.ask_rules = []
        app._permission_engine.deny_rules = []
        mode = app._get_permission_mode()
        assert mode == "yolo"

    def test_strict_mode_detected(self) -> None:
        app = self.make_app("normal")
        deny_rule = MagicMock()
        deny_rule.pattern = "Shell(*)"
        app._permission_engine.deny_rules = [deny_rule]
        mode = app._get_permission_mode()
        assert mode == "strict"

    def test_no_permission_engine_returns_normal(self) -> None:
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=None,
            audit_trail=None,
            session_id="test",
        )
        assert app._get_permission_mode() == "normal"

    def test_get_permission_mode_strict_high_deny_count(self) -> None:
        deny_rule = MagicMock()
        deny_rule.pattern = "FileWrite(*)"
        perm_engine = MagicMock()
        perm_engine.deny_rules = [deny_rule]
        perm_engine.ask_rules = [MagicMock()]
        perm_engine.session_grants = []
        perm_engine.plan_mode = False
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        assert app._get_permission_mode() == "strict"

    def test_get_permission_mode_strict_many_deny_rules(self) -> None:
        deny_rules = [MagicMock() for _ in range(6)]
        perm_engine = MagicMock()
        perm_engine.deny_rules = deny_rules
        perm_engine.ask_rules = [MagicMock()]
        perm_engine.session_grants = []
        perm_engine.plan_mode = False
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        assert app._get_permission_mode() == "strict"

    def test_get_permission_mode_yolo(self) -> None:
        perm_engine = MagicMock()
        perm_engine.deny_rules = []
        perm_engine.ask_rules = []
        perm_engine.session_grants = []
        perm_engine.plan_mode = False
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        assert app._get_permission_mode() == "yolo"

    def test_get_permission_mode_plan(self) -> None:
        perm_engine = MagicMock()
        perm_engine.deny_rules = []
        perm_engine.ask_rules = [MagicMock()]
        perm_engine.session_grants = []
        perm_engine.plan_mode = True
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        assert app._get_permission_mode() == "plan"


class TestGetPromptState:
    def test_plan_mode_returns_plan(self) -> None:
        perm_engine = MagicMock()
        perm_engine.plan_mode = True
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        app._pause_event = asyncio.Event()
        app._pause_event.set()
        assert app._get_prompt_state() == "plan"

    def test_paused_returns_paused(self) -> None:
        perm_engine = MagicMock()
        perm_engine.plan_mode = False
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        app._pause_event = asyncio.Event()
        app._pause_event.clear()
        assert app._get_prompt_state() == "paused"

    def test_normal_returns_empty(self) -> None:
        perm_engine = MagicMock()
        perm_engine.plan_mode = False
        app = TUIApp(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            tool_context=MagicMock(),
            conversation=MagicMock(),
            permission_engine=perm_engine,
            audit_trail=None,
            session_id="test",
        )
        app._pause_event = asyncio.Event()
        app._pause_event.set()
        assert app._get_prompt_state() == ""


class TestThinkingSpinner:
    async def test_start_stop(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        sm.start()
        assert sm._started
        sm.stop()
        assert not sm._started

    async def test_start_twice_noop(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        sm.start()
        sm.start()
        assert sm._started

    def test_update_before_start_noop(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        sm.update("shell", {"command": "echo hi"})

    def test_stop_without_start_noop(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        sm.stop()

    async def test_update_shows_tool_name(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        with patch("godspeed.tui.app.Status") as mock_status:
            sm.start()
            sm.update("file_read", {"file_path": "/some/path/file.txt"})
            mock_status.return_value.update.assert_called()

    async def test_update_truncates_long_path(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        with patch("godspeed.tui.app.Status"):
            sm.start()
            long_path = "x" * 100
            sm.update("file_read", {"file_path": long_path})
            assert len(sm._tool_label) < len(long_path)

    async def test_wrap_stops_spinner(self) -> None:
        from godspeed.tui.app import _ThinkingSpinner

        sm = _ThinkingSpinner()
        with patch("godspeed.tui.app.Status"):
            sm.start()
            fn = MagicMock(return_value=42)
            wrapped = sm.wrap(fn)
            result = wrapped("arg1", kw="val")
            assert result == 42
            assert not sm._started


class TestBuildKeyBindings:
    def test_key_bindings_created(self) -> None:
        from godspeed.tui.app import _build_key_bindings

        bindings = _build_key_bindings()
        assert bindings is not None


class TestOnSigint:
    def test_first_sigint_sets_cancel(self) -> None:
        llm_client = MagicMock()
        tool_registry = MagicMock()
        tool_context = MagicMock()
        conversation = MagicMock()
        permission_engine = MagicMock()
        permission_engine.deny_rules = []
        permission_engine.ask_rules = [MagicMock()]
        permission_engine.session_grants = []
        permission_engine.plan_mode = False

        app = TUIApp(
            llm_client=llm_client,
            tool_registry=tool_registry,
            tool_context=tool_context,
            conversation=conversation,
            permission_engine=permission_engine,
            audit_trail=None,
            session_id="test",
        )
        # First SIGINT sets cancel_event but does not raise
        app._on_sigint()
        assert app._cancel_event.is_set()

    def test_second_sigint_within_1s_raises(self) -> None:
        import time

        llm_client = MagicMock()
        tool_registry = MagicMock()
        tool_context = MagicMock()
        conversation = MagicMock()
        permission_engine = MagicMock()
        permission_engine.deny_rules = []
        permission_engine.ask_rules = [MagicMock()]
        permission_engine.session_grants = []
        permission_engine.plan_mode = False

        app = TUIApp(
            llm_client=llm_client,
            tool_registry=tool_registry,
            tool_context=tool_context,
            conversation=conversation,
            permission_engine=permission_engine,
            audit_trail=None,
            session_id="test",
        )
        # Set cancel_event (as if first SIGINT was already received)
        app._cancel_event.set()
        app._last_sigint_monotonic = time.monotonic()
        # Second press within 1s must raise
        with pytest.raises(KeyboardInterrupt):
            app._on_sigint()


class TestSuggestionAutoPermission:
    def test_skip_if_already_in_allow_rules(self) -> None:
        engine = MagicMock()
        engine.allow_rules = ["FileWrite(*)"]
        engine.deny_rules = []
        engine.ask_rules = []
        proxy = _InteractivePermissionProxy(engine)
        # Should return early without printing
        with patch("godspeed.tui.app._output.console.print") as mock_print:
            proxy._suggest_auto_permission("FileWrite(*)")
            mock_print.assert_not_called()

    def test_suggest_and_confirm(self) -> None:
        engine = MagicMock()
        engine.allow_rules = []
        engine.deny_rules = []
        engine.ask_rules = []
        proxy = _InteractivePermissionProxy(engine)
        with (
            patch("godspeed.tui.app._output.console.print") as mock_print,
            patch("godspeed.tui.app._output.console.input", return_value="y"),
            patch("godspeed.config.append_allow_rule", return_value=True),
        ):
            proxy._suggest_auto_permission("Shell(*)")
            # Should print suggestion and call append
            assert mock_print.called

    def test_suggest_and_decline(self) -> None:
        engine = MagicMock()
        engine.allow_rules = []
        engine.deny_rules = []
        engine.ask_rules = []
        proxy = _InteractivePermissionProxy(engine)
        with (
            patch("godspeed.tui.app._output.console.print") as mock_print,
            patch("godspeed.tui.app._output.console.input", return_value="n"),
        ):
            proxy._suggest_auto_permission("Shell(*)")
            assert mock_print.called

    def test_suggest_keyboard_interrupt(self) -> None:
        engine = MagicMock()
        engine.allow_rules = []
        engine.deny_rules = []
        engine.ask_rules = []
        proxy = _InteractivePermissionProxy(engine)
        with (
            patch("godspeed.tui.app._output.console.input", side_effect=KeyboardInterrupt),
            patch("godspeed.tui.app._output.console.print"),
        ):
            proxy._suggest_auto_permission("Shell(*)")
            # Should handle gracefully
