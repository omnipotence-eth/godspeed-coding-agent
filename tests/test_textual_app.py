"""Tests for the simplified Textual-based TUI."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button

from godspeed.tui.textual_app import (
    ChatPanel,
    CommandPaletteScreen,
    DiffReviewScreen,
    GodspeedTextualApp,
    InfoPanel,
    InputBar,
    PermissionScreen,
    StreamingIndicator,
    ToolCallBlock,
    UserMessage,
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
    async def test_info_panel_reactive(self):
        panel = InfoPanel()
        async with App().run_test() as pilot:
            await pilot.app.mount(panel)
            await pilot.pause()
            panel.session_id = "abc123"
            panel.model = "gpt-4"
            panel.project_dir = "C:\\Users\\test"
            panel.input_tokens = 100
            panel.output_tokens = 200
            panel.context_pct = 42.0
            panel.cost_usd = 0.05
            panel.tool_calls = 5
            panel.tool_errors = 1
            panel.tool_denied = 0
            panel.permission_mode = "strict"
            panel.is_running = True
            assert panel.session_id == "abc123"

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


class TestAppCompose:
    """Integration-ish tests using Textual Pilot."""

    @pytest.mark.asyncio
    async def test_app_composes(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#chat-panel", ChatPanel) is not None
            assert app.query_one("#info-panel", InfoPanel) is not None
            assert app.query_one("#input-bar", InputBar) is not None

    @pytest.mark.asyncio
    async def test_info_panel_reactive_updates(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            panel = app.query_one("#info-panel", InfoPanel)
            panel.model = "gpt-4o"
            await pilot.pause()
            assert panel.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_dispatch_command_help(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._dispatch_command("/help")
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

    @pytest.mark.asyncio
    async def test_action_clear_chat(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chat-panel", ChatPanel)
            chat.write_user("test")
            app.action_clear_chat()
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_command_palette_opens(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            assert len(app.screen_stack) > 1
