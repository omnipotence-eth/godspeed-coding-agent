"""Tests for the Textual-based TUI.

Uses Textual's ``Pilot`` to exercise widget composition and
screen transitions without a real terminal.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App

from godspeed.tui.textual_app import (
    ChatPanel,
    ContextPanel,
    DiffReviewScreen,
    GodspeedTextualApp,
    HeaderBar,
    InputBar,
    PermissionScreen,
    Sidebar,
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Fake tool {name}"


class _FakeRegistry:
    def list_tools(self):
        return [_FakeTool("shell"), _FakeTool("file_read")]

    def get(self, name: str):
        for t in self.list_tools():
            if t.name == name:
                return t
        return None


@pytest.fixture
def fake_registry():
    return _FakeRegistry()


@pytest.fixture
def mock_deps():
    """Return minimal mocked dependencies for GodspeedTextualApp."""
    llm = MagicMock()
    llm.model = "gpt-4"
    llm.total_cost_usd = 0.0
    llm.total_input_tokens = 0
    llm.total_output_tokens = 0

    tool_ctx = MagicMock()
    tool_ctx.cwd = "C:\\Users\\test\\project"
    tool_ctx.permissions = None
    tool_ctx.diff_reviewer = None

    conv = MagicMock()
    conv.token_count = 100
    conv.max_tokens = 4096

    return {
        "llm_client": llm,
        "tool_registry": _FakeRegistry(),
        "tool_context": tool_ctx,
        "conversation": conv,
        "permission_engine": None,
        "audit_trail": None,
        "session_id": "test-session",
    }


class TestWidgets:
    """Unit tests for individual widgets."""

    @pytest.mark.asyncio
    async def test_header_bar_updates(self):
        bar = HeaderBar()
        async with App().run_test() as pilot:
            await pilot.app.mount(bar)
            await pilot.pause()
            bar.model = "claude-sonnet"
            bar.cost_usd = 0.1234
            bar.turns = 5
            bar.permission_mode = "strict"
            bar.context_pct = 42.0
            bar.input_tokens = 100
            bar.output_tokens = 200
            assert bar.model == "claude-sonnet"

    @pytest.mark.asyncio
    async def test_chat_panel_write(self):
        panel = ChatPanel()
        async with App().run_test() as pilot:
            await pilot.app.mount(panel)
            await pilot.pause()
            panel.write("hello")
            panel.write_user("user msg")
            panel.write_assistant("assistant msg")
            panel.write_tool_call("shell", {"command": "ls"})
            panel.write_tool_result("shell", "output")
            panel.write_error("boom")
            panel.write_system("info")

    @pytest.mark.asyncio
    async def test_context_panel_update(self):
        panel = ContextPanel()
        async with App().run_test() as pilot:
            await pilot.app.mount(panel)
            await pilot.pause()
            panel.update_session_info("foo\nbar")

    @pytest.mark.asyncio
    async def test_input_bar_get_clear(self):
        bar = InputBar()
        async with App().run_test() as pilot:
            await pilot.app.mount(bar)
            await pilot.pause()
            inp = bar.query_one("#user-input")
            inp.value = "hello"
            assert bar.get_value() == "hello"
            bar.clear()
            assert bar.get_value() == ""

    @pytest.mark.asyncio
    async def test_sidebar_lists_tools(self, fake_registry):
        sidebar = Sidebar(fake_registry)
        async with App().run_test() as pilot:
            await pilot.app.mount(sidebar)
            await pilot.pause()


class TestScreens:
    """Unit tests for modal screens."""

    @pytest.mark.asyncio
    async def test_permission_screen_dismiss(self):
        screen = PermissionScreen(
            tool_name="shell",
            reason="destructive",
            arguments={"command": "rm -rf /"},
        )
        async with App().run_test() as pilot:
            await pilot.app.push_screen(screen)
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_diff_review_screen_dismiss(self):
        screen = DiffReviewScreen(
            tool_name="file_edit",
            path="C:\\Users\\test\\project\\foo.py",
            before="old",
            after="new",
        )
        async with App().run_test() as pilot:
            await pilot.app.push_screen(screen)
            await pilot.pause()


class TestAppCompose:
    """Integration-ish tests using Textual Pilot."""

    @pytest.mark.asyncio
    async def test_app_composes(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            # All major widgets should be mountable
            assert app.query_one("#header", HeaderBar) is not None
            assert app.query_one("#chat", ChatPanel) is not None
            assert app.query_one("#context", ContextPanel) is not None
            assert app.query_one("#input", InputBar) is not None

    @pytest.mark.asyncio
    async def test_header_reactive_updates(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app.query_one("#header", HeaderBar)
            header.model = "gpt-4o"
            header.turns = 3
            await pilot.pause()
            assert header.model == "gpt-4o"
            assert header.turns == 3

    @pytest.mark.asyncio
    async def test_dispatch_command_help(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._dispatch_command("/help")
            # Should not crash; verify turn count unchanged
            assert app.turn_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_command_unknown(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._dispatch_command("/unknown")
            assert app.turn_count == 0

    @pytest.mark.asyncio
    async def test_input_triggers_agent(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#user-input")
            inp.value = "hello"
            # Submit via the input widget
            await pilot.press("enter")
            await pilot.pause()
            assert app.turn_count == 1

    @pytest.mark.asyncio
    async def test_empty_input_ignored(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#user-input")
            inp.value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert app.turn_count == 0
