"""Tests for parallel tool execution in the agent loop."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_multi_tool_response(
    tools: list[tuple[str, dict[str, Any]]],
) -> ChatResponse:
    """Create an LLM response with multiple tool calls."""
    tool_calls = []
    for i, (name, args) in enumerate(tools):
        tool_calls.append(
            {
                "id": f"call_{i:03d}",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        )
    return ChatResponse(
        content="",
        tool_calls=tool_calls,
        finish_reason="tool_calls",
    )


class SlowTool(Tool):
    """A tool that takes a configurable amount of time to execute."""

    def __init__(
        self,
        name: str = "slow_tool",
        delay: float = 0.1,
        result: ToolResult | None = None,
    ) -> None:
        self._name = name
        self._delay = delay
        self._result = result or ToolResult.success(f"{name} output")
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"A slow tool ({self._delay}s)"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return self._result


class TestParallelToolExecution:
    """Test parallel tool dispatch via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_multiple_tools_execute_concurrently(self, tool_context) -> None:
        """Two slow tools should complete faster than their combined delay."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        tool_a = SlowTool(name="tool_a", delay=0.5)
        tool_b = SlowTool(name="tool_b", delay=0.5)
        registry.register(tool_a)
        registry.register(tool_b)

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response(
                    [
                        ("tool_a", {"input": "a"}),
                        ("tool_b", {"input": "b"}),
                    ]
                ),
                _make_text_response("Done"),
            ]
        )

        t0 = time.monotonic()
        result = await agent_loop(
            "Run both",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=True,
        )
        elapsed = time.monotonic() - t0

        assert result == "Done"
        assert tool_a.call_count == 1
        assert tool_b.call_count == 1
        # Parallel: both 0.5s tools should overlap. Sequential would be ~1.0s+.
        # With overhead, parallel should still be well under 0.9s.
        assert elapsed < 0.9, f"Expected parallel execution, took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_results_feed_back_in_order(self, tool_context) -> None:
        """Tool results must appear in conversation in the same order as calls."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # Tool B is faster than A, but results should be ordered A, B, C
        registry.register(SlowTool(name="tool_a", delay=0.1, result=ToolResult.success("result_a")))
        registry.register(
            SlowTool(name="tool_b", delay=0.01, result=ToolResult.success("result_b"))
        )
        registry.register(
            SlowTool(name="tool_c", delay=0.05, result=ToolResult.success("result_c"))
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response(
                    [
                        ("tool_a", {}),
                        ("tool_b", {}),
                        ("tool_c", {}),
                    ]
                ),
                _make_text_response("All done"),
            ]
        )

        await agent_loop(
            "Run all three",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=True,
        )

        # Extract tool results from conversation in order
        tool_results = [
            msg["content"]
            for msg in conversation.messages
            if msg.get("role") == "tool" and msg.get("content", "").startswith("result_")
        ]
        assert tool_results == ["result_a", "result_b", "result_c"]

    @pytest.mark.asyncio
    async def test_permission_denied_skips_without_blocking_others(self, tool_context) -> None:
        """One denied tool should not prevent others from executing."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="tool_a", result=ToolResult.success("a_ok")))
        registry.register(MockTool(name="tool_b", result=ToolResult.success("b_ok")))

        # Set up permissions to deny tool_b
        class DenyBPermissions:
            def evaluate(self, tool_call):
                if tool_call.tool_name == "tool_b":
                    return "deny"
                return "allow"

        tool_context_with_perms = ToolContext(
            cwd=tool_context.cwd,
            session_id=tool_context.session_id,
            permissions=DenyBPermissions(),
        )

        denied_tools = []

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response(
                    [
                        ("tool_a", {}),
                        ("tool_b", {}),
                    ]
                ),
                _make_text_response("Done with A only"),
            ]
        )

        result = await agent_loop(
            "Run both",
            conversation,
            client,
            registry,
            tool_context_with_perms,
            on_permission_denied=lambda name, reason: denied_tools.append(name),
            parallel_tool_calls=True,
        )

        assert result == "Done with A only"
        assert "tool_b" in denied_tools

        # tool_a should have executed, tool_b should be denied
        tool_msgs = [msg for msg in conversation.messages if msg.get("role") == "tool"]
        contents = [m["content"] for m in tool_msgs]
        assert any("a_ok" in c for c in contents)
        assert any("DENIED" in c for c in contents)

    @pytest.mark.asyncio
    async def test_sequential_fallback_when_disabled(self, tool_context) -> None:
        """With parallel_tool_calls=False, tools execute sequentially."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        execution_order: list[str] = []

        class OrderTrackingTool(Tool):
            def __init__(self, tool_name: str, delay: float) -> None:
                self._name = tool_name
                self._delay = delay

            @property
            def name(self) -> str:
                return self._name

            @property
            def description(self) -> str:
                return "Tracks execution order"

            @property
            def risk_level(self) -> RiskLevel:
                return RiskLevel.READ_ONLY

            def get_schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}, "required": []}

            async def execute(self, arguments, context) -> ToolResult:
                execution_order.append(self._name)
                await asyncio.sleep(self._delay)
                return ToolResult.success(f"{self._name} done")

        registry.register(OrderTrackingTool("first", 0.05))
        registry.register(OrderTrackingTool("second", 0.05))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response([("first", {}), ("second", {})]),
                _make_text_response("Done"),
            ]
        )

        t0 = time.monotonic()
        await agent_loop(
            "Run both",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=False,
        )
        elapsed = time.monotonic() - t0

        assert execution_order == ["first", "second"]
        # Sequential: should take >= 0.10s (two 0.05s delays)
        assert elapsed >= 0.09, f"Expected sequential execution, took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_single_tool_call_unchanged(self, tool_context) -> None:
        """A single tool call should work identically regardless of parallel flag."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        tool = MockTool(name="file_read", result=ToolResult.success("file contents"))
        registry.register(tool)

        client = LLMClient(model="test")

        single_response = ChatResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_001",
                    "function": {
                        "name": "file_read",
                        "arguments": json.dumps({"file_path": "test.py"}),
                    },
                }
            ],
            finish_reason="tool_calls",
        )
        client.chat = AsyncMock(side_effect=[single_response, _make_text_response("Read it")])

        result = await agent_loop(
            "Read test.py",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=True,
        )
        assert result == "Read it"
        assert tool.last_arguments == {"file_path": "test.py"}

    @pytest.mark.asyncio
    async def test_callbacks_called_for_all_parallel_tools(self, tool_context) -> None:
        """on_tool_call and on_tool_result should fire for every tool in a parallel batch."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="tool_a", result=ToolResult.success("a_out")))
        registry.register(MockTool(name="tool_b", result=ToolResult.success("b_out")))

        tool_call_names: list[str] = []
        tool_result_names: list[str] = []

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response([("tool_a", {}), ("tool_b", {})]),
                _make_text_response("Done"),
            ]
        )

        await agent_loop(
            "Run both",
            conversation,
            client,
            registry,
            tool_context,
            on_tool_call=lambda name, args: tool_call_names.append(name),
            on_tool_result=lambda name, result: tool_result_names.append(name),
            parallel_tool_calls=True,
        )

        assert set(tool_call_names) == {"tool_a", "tool_b"}
        assert set(tool_result_names) == {"tool_a", "tool_b"}

    @pytest.mark.asyncio
    async def test_parallel_with_one_error_tool(self, tool_context) -> None:
        """One failing tool in a parallel batch should not prevent others from succeeding."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="good_tool", result=ToolResult.success("ok")))
        registry.register(MockTool(name="bad_tool", result=ToolResult.failure("boom")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response(
                    [
                        ("good_tool", {}),
                        ("bad_tool", {}),
                    ]
                ),
                _make_text_response("Handled the error"),
            ]
        )

        result = await agent_loop(
            "Run both",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=True,
        )

        assert result == "Handled the error"

        # Both results should be in conversation
        tool_msgs = [msg["content"] for msg in conversation.messages if msg.get("role") == "tool"]
        assert any("ok" in c for c in tool_msgs)
        assert any("boom" in c for c in tool_msgs)

    @pytest.mark.asyncio
    async def test_all_denied_continues_to_next_llm_turn(self, tool_context) -> None:
        """If all tool calls are denied, the loop should continue to the next LLM turn."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="tool_a"))

        class DenyAllPermissions:
            def evaluate(self, tool_call):
                return "deny"

        tool_context_deny = ToolContext(
            cwd=tool_context.cwd,
            session_id=tool_context.session_id,
            permissions=DenyAllPermissions(),
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_multi_tool_response([("tool_a", {})]),
                _make_text_response("Everything was denied, giving up."),
            ]
        )

        result = await agent_loop(
            "Try it",
            conversation,
            client,
            registry,
            tool_context_deny,
            parallel_tool_calls=True,
        )
        assert "denied" in result.lower() or "giving up" in result.lower()
