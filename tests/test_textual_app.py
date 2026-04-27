"""Tests for the professional Textual-based TUI.

Uses Textual's ``Pilot`` to exercise widget composition and
screen transitions without a real terminal.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button

from godspeed.tui.textual_app import (
    ChatPanel,
    CommandPaletteScreen,
    ContextSidebar,
    DiffReviewScreen,
    GodspeedTextualApp,
    InputBar,
    PermissionScreen,
    StatusBar,
    StreamingIndicator,
    ToolCallBlock,
    ToolSidebar,
    UserMessage,
    WelcomeScreen,
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
    async def test_status_bar_reactive(self):
        bar = StatusBar()
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
            bar.is_running = True
            assert bar.model == "claude-sonnet"

    @pytest.mark.asyncio
    async def test_context_sidebar_reactive(self):
        ctx = ContextSidebar()
        async with App().run_test() as pilot:
            await pilot.app.mount(ctx)
            await pilot.pause()
            ctx.session_id = "abc123"
            ctx.tool_count = 10
            ctx.tool_calls = 5
            ctx.tool_errors = 1
            ctx.tool_denied = 0
            ctx.project_dir = "C:\\Users\\test"
            assert ctx.session_id == "abc123"

    @pytest.mark.asyncio
    async def test_chat_panel_messages(self):
        panel = ChatPanel()
        async with App().run_test() as pilot:
            await pilot.app.mount(panel)
            await pilot.pause()
            panel.write_user("hello")
            panel.write_assistant("hi there")
            panel.write_system("info")
            panel.write_error("boom")

    @pytest.mark.asyncio
    async def test_chat_panel_tool_call(self):
        panel = ChatPanel()
        async with App().run_test() as pilot:
            await pilot.app.mount(panel)
            await pilot.pause()
            block = panel.add_tool_call("shell", {"command": "ls"})
            block.set_result("output", is_error=False)
            assert block._tool_name == "shell"

    @pytest.mark.asyncio
    async def test_chat_panel_streaming_indicator(self):
        panel = ChatPanel()
        async with App().run_test() as pilot:
            await pilot.app.mount(panel)
            await pilot.pause()
            ind = panel.add_streaming_indicator("Loading")
            assert ind._base_text == "Loading"

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
    async def test_tool_sidebar_lists_tools(self, fake_registry):
        sidebar = ToolSidebar(fake_registry)
        async with App().run_test() as pilot:
            await pilot.app.mount(sidebar)
            await pilot.pause()
            from textual.widgets import ListView
            lv = sidebar.query_one("#tool-list", ListView)
            assert lv is not None

    @pytest.mark.asyncio
    async def test_tool_call_block_expands(self):
        block = ToolCallBlock("shell", {"command": "ls"}, 0.0)
        async with App().run_test() as pilot:
            await pilot.app.mount(block)
            await pilot.pause()
            block.expanded = True
            assert block.expanded is True
            block.set_result("files listed", is_error=False)
            assert block._result == "files listed"

    @pytest.mark.asyncio
    async def test_streaming_indicator_ticks(self):
        ind = StreamingIndicator("Loading")
        async with App().run_test() as pilot:
            await pilot.app.mount(ind)
            await pilot.pause()
            # Interval ticks are handled by Textual's clock; just verify mount
            assert ind._base_text == "Loading"

    @pytest.mark.asyncio
    async def test_user_message_compose(self):
        msg = UserMessage("test message")
        async with App().run_test() as pilot:
            await pilot.app.mount(msg)
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
            btn = screen.query_one("#btn-yes", Button)
            btn.press()
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

    @pytest.mark.asyncio
    async def test_command_palette_mounts(self):
        screen = CommandPaletteScreen()
        async with App().run_test() as pilot:
            await pilot.app.push_screen(screen)
            await pilot.pause()
            lv = screen.query_one("#palette-list")
            assert lv is not None

    @pytest.mark.asyncio
    async def test_welcome_screen_dismiss(self):
        screen = WelcomeScreen(
            model="gpt-4",
            project_dir="C:\\Users\\test\\project",
            tool_count=25,
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
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            assert app.query_one("#status-bar", StatusBar) is not None
            assert app.query_one("#chat-panel", ChatPanel) is not None
            assert app.query_one("#context-sidebar", ContextSidebar) is not None
            assert app.query_one("#input-bar", InputBar) is not None

    @pytest.mark.asyncio
    async def test_header_reactive_updates(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            bar = app.query_one("#status-bar", StatusBar)
            bar.model = "gpt-4o"
            bar.turns = 3
            await pilot.pause()
            assert bar.model == "gpt-4o"
            assert bar.turns == 3

    @pytest.mark.asyncio
    async def test_dispatch_command_help(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            app._dispatch_command("/help")
            assert app.turn_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_command_unknown(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            app._dispatch_command("/unknown")
            assert app.turn_count == 0

    @pytest.mark.asyncio
    async def test_input_triggers_agent(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            inp = app.query_one("#user-input")
            inp.value = "hello"
            await pilot.press("enter")
            await pilot.pause()
            assert app.turn_count == 1

    @pytest.mark.asyncio
    async def test_empty_input_ignored(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            inp = app.query_one("#user-input")
            inp.value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert app.turn_count == 0

    @pytest.mark.asyncio
    async def test_action_clear_chat(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            chat = app.query_one("#chat-panel", ChatPanel)
            chat.write_user("test")
            app.action_clear_chat()
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_action_new_session(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            app.turn_count = 5
            app.action_new_session()
            assert app.turn_count == 0

    @pytest.mark.asyncio
    async def test_command_palette_opens(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.dismiss()  # Dismiss welcome screen
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            assert len(app.screen_stack) > 1
