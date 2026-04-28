"""Tests for the minimal Textual TUI."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.widgets import Button

from godspeed.tui.textual_app import (
    CommandPaletteScreen,
    GodspeedTextualApp,
    PermissionScreen,
)


class _FakeRegistry:
    def list_tools(self):
        from types import SimpleNamespace
        return [SimpleNamespace(name="shell", description="shell")]


@pytest.fixture
def mock_deps():
    llm = MagicMock()
    llm.model = "gpt-4"
    llm.total_cost_usd = 0.0
    llm.total_input_tokens = 0
    llm.total_output_tokens = 0

    tool_ctx = MagicMock()
    tool_ctx.cwd = "C:\\test"
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
        "session_id": "test",
    }


class TestApp:
    @pytest.mark.asyncio
    async def test_composes(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#chat-log") is not None
            assert app.query_one("#user-input") is not None
            assert app.query_one("#info-session") is not None

    @pytest.mark.asyncio
    async def test_input_focused(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#user-input")
            assert inp.has_focus

    @pytest.mark.asyncio
    async def test_send_message(self, mock_deps):
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
    async def test_command_palette(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            assert len(app.screen_stack) > 1

    @pytest.mark.asyncio
    async def test_permission_screen(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = PermissionScreen(tool_name="shell", reason="test")
            await pilot.app.push_screen(screen)
            await pilot.pause()
            btn = screen.query_one("#btn-yes", Button)
            btn.press()
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_palette_screen(self, mock_deps):
        app = GodspeedTextualApp(**mock_deps)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = CommandPaletteScreen()
            await pilot.app.push_screen(screen)
            await pilot.pause()
            lv = screen.query_one("#palette-list")
            assert lv is not None
