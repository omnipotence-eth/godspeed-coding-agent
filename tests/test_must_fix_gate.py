"""Tests for the MUST-FIX gate on auto-verify failures (v2.5.1).

When auto-verify leaves unresolved lint errors, the loop must inject a
user-role message forcing the agent to fix them before any other edits.
Capped at MUST_FIX_CAP injections per session.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import MUST_FIX_CAP, _maybe_inject_must_fix, agent_loop
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _text_resp(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _tool_resp(name: str, args: dict[str, Any], call_id: str = "call_1") -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[{"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}],
        finish_reason="tool_calls",
    )


def _count_must_fix_messages(conversation: Conversation) -> int:
    return sum(
        1
        for msg in conversation.messages
        if msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and "MUST fix" in msg["content"]
    )


class TestMaybeInjectMustFix:
    """Direct unit tests of the gate helper — no agent loop."""

    def test_no_injection_on_clean_verify(self) -> None:
        convo = Conversation("sys", max_tokens=100_000)
        out = _maybe_inject_must_fix(convo, "x.py", "Verification passed: x.py", 0)
        assert out == 0
        assert _count_must_fix_messages(convo) == 0

    def test_injects_on_some_remaining_fingerprint(self) -> None:
        convo = Conversation("sys", max_tokens=100_000)
        verify_output = (
            "Auto-fixed 2 round(s) of issues, some remaining: app.py\nF401 unused import"
        )
        out = _maybe_inject_must_fix(convo, "app.py", verify_output, 0)
        assert out == 1
        assert _count_must_fix_messages(convo) == 1
        must_fix = next(m for m in convo.messages if "MUST fix" in str(m.get("content", "")))
        assert "app.py" in must_fix["content"]
        assert "F401" in must_fix["content"]

    def test_caps_at_must_fix_cap(self) -> None:
        convo = Conversation("sys", max_tokens=100_000)
        verify_output = "Auto-fixed rounds, some remaining: app.py"
        counter = 0
        for _ in range(MUST_FIX_CAP + 2):
            counter = _maybe_inject_must_fix(convo, "app.py", verify_output, counter)
        assert counter == MUST_FIX_CAP
        assert _count_must_fix_messages(convo) == MUST_FIX_CAP

    def test_empty_output_never_injects(self) -> None:
        convo = Conversation("sys", max_tokens=100_000)
        assert _maybe_inject_must_fix(convo, "x.py", "", 0) == 0
        assert _maybe_inject_must_fix(convo, "x.py", None, 0) == 0  # type: ignore[arg-type]


class TestMustFixGateInAgentLoop:
    """End-to-end: the loop fires the gate after auto-verify leaves errors."""

    @pytest.mark.asyncio
    async def test_gate_fires_after_auto_verify_leaves_errors_sequential(
        self, tool_context
    ) -> None:
        """A single file_edit → verify-with-remaining-errors → MUST-FIX injected."""
        convo = Conversation("sys", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="verify"))  # presence required by the gate check

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_resp("file_edit", {"file_path": "app.py"}),
                _text_resp("done"),
            ]
        )

        dirty = ToolResult.success("Auto-fixed 1 round(s) of issues, some remaining: app.py\nF401")
        with patch("godspeed.agent.loop._auto_verify_file", AsyncMock(return_value=dirty)):
            await agent_loop(
                "edit app.py", convo, client, registry, tool_context, parallel_tool_calls=False
            )

        assert _count_must_fix_messages(convo) == 1

    @pytest.mark.asyncio
    async def test_gate_fires_in_parallel_path(self, tool_context) -> None:
        """Two concurrent file_edit calls with failing verify → two injections."""
        convo = Conversation("sys", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="verify"))

        client = LLMClient(model="test")
        # Two tool calls in one response → triggers parallel path (len > 1).
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "function": {
                                "name": "file_edit",
                                "arguments": json.dumps({"file_path": "a.py"}),
                            },
                        },
                        {
                            "id": "c2",
                            "function": {
                                "name": "file_edit",
                                "arguments": json.dumps({"file_path": "b.py"}),
                            },
                        },
                    ],
                    finish_reason="tool_calls",
                ),
                _text_resp("done"),
            ]
        )

        dirty = ToolResult.success("Auto-fixed 1 round(s) of issues, some remaining: app.py\nF401")
        with patch("godspeed.agent.loop._auto_verify_file", AsyncMock(return_value=dirty)):
            await agent_loop(
                "edit both", convo, client, registry, tool_context, parallel_tool_calls=True
            )

        assert _count_must_fix_messages(convo) == 2

    @pytest.mark.asyncio
    async def test_no_injection_when_verify_passes(self, tool_context) -> None:
        convo = Conversation("sys", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="verify"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_resp("file_edit", {"file_path": "app.py"}),
                _text_resp("done"),
            ]
        )

        clean = ToolResult.success("Verification passed: app.py")
        with patch("godspeed.agent.loop._auto_verify_file", AsyncMock(return_value=clean)):
            await agent_loop(
                "edit app.py", convo, client, registry, tool_context, parallel_tool_calls=False
            )

        assert _count_must_fix_messages(convo) == 0
