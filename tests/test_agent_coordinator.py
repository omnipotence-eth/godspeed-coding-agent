"""Tests for the sub-agent coordinator.

v3.5: coordinator switched from a depth-counter to structural recursion
prevention (child registry without ``spawn_agent``) plus per-spawn
timeouts. Tests exercise the new semantics.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.coordinator import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TIMEOUT_SECONDS,
    AgentCoordinator,
    SpawnAgentTool,
)
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_tool_response(tool_name: str, arguments: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[
            {
                "id": "call_sub_001",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
        finish_reason="tool_calls",
    )


class TestAgentCoordinator:
    """Sub-agent spawning, isolation, and recursion prevention."""

    @pytest.mark.asyncio
    async def test_spawn_returns_result(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=_make_text_response("Sub-task completed successfully.")
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Find all TODO comments")
        assert "completed successfully" in result

    @pytest.mark.asyncio
    async def test_spawn_uses_tools(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        registry.register(
            MockTool(name="grep_search", result=ToolResult.success("TODO found in main.py:42"))
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("grep_search", {"pattern": "TODO"}),
                _make_text_response("Found 1 TODO in main.py line 42."),
            ]
        )

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Search for TODOs")
        assert "TODO" in result
        assert client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_isolated_conversation(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        r1 = await coordinator.spawn("Task A")
        r2 = await coordinator.spawn("Task B")

        assert r1 == "Done."
        assert r2 == "Done."
        assert client.chat.call_count == 2

        # Each spawn starts a fresh conversation — Task A's text must
        # NOT appear in Task B's request messages.
        second_call_messages = client.chat.call_args_list[1][1]["messages"]
        user_messages = [m for m in second_call_messages if m.get("role") == "user"]
        assert any("Task B" in m["content"] for m in user_messages)
        assert not any("Task A" in m.get("content", "") for m in user_messages)

    @pytest.mark.asyncio
    async def test_child_registry_excludes_spawn_agent(self, tool_context: ToolContext) -> None:
        # The structural recursion guard: the child registry must
        # NOT contain a spawn_agent tool, regardless of what the
        # parent registered.
        registry = ToolRegistry()

        # Fake a spawn_agent registration on the parent — coordinator
        # should filter it out in the child.
        class _FakeSpawn(MockTool):
            pass

        registry.register(MockTool(name="spawn_agent", result=ToolResult.success("nested")))
        registry.register(MockTool(name="file_read", result=ToolResult.success("contents")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("No recursion."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )
        child_reg = coordinator._build_child_registry()
        assert child_reg.get("spawn_agent") is None
        assert child_reg.get("file_read") is not None

    @pytest.mark.asyncio
    async def test_iteration_limit_respected(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_tool_response("shell", {"command": "echo loop"}))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
            default_max_iterations=3,
        )

        result = await coordinator.spawn("Loop forever")
        assert "maximum iterations" in result.lower()
        assert client.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_per_spawn_max_iterations_override(self, tool_context: ToolContext) -> None:
        # The per-call max_iterations argument should clamp below the
        # coordinator default when passed explicitly.
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_tool_response("shell", {"command": "echo"}))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
            default_max_iterations=DEFAULT_MAX_ITERATIONS,
        )

        await coordinator.spawn("Loop", max_iterations=2)
        assert client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_failure_doesnt_crash(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        result = await coordinator.spawn("Risky task")
        # The failure is surfaced as an error string, never as an
        # exception to the parent.
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_spawn_parallel(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Parallel task done."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )

        results = await coordinator.spawn_parallel(["Task 1", "Task 2", "Task 3"])
        assert len(results) == 3
        assert all("done" in r.lower() for r in results)


class TestSpawnAgentTool:
    """Tool adapter for LLM-issued spawns."""

    def test_metadata(self) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)
        assert tool.name == "spawn_agent"
        # HIGH so the user sees a permission prompt for each spawn.
        assert tool.risk_level == "high"
        schema = tool.get_schema()
        assert "task" in schema["properties"]
        assert "max_iterations" in schema["properties"]
        assert "timeout" in schema["properties"]
        assert schema["required"] == ["task"]

    @pytest.mark.asyncio
    async def test_execute_delegates_to_coordinator(self, tool_context: ToolContext) -> None:
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Sub-agent result."))

        coordinator = AgentCoordinator(
            llm_client=client,
            tool_registry=registry,
            tool_context=tool_context,
        )
        tool = SpawnAgentTool(coordinator)

        result = await tool.execute({"task": "Do something"}, tool_context)
        assert not result.is_error
        assert "Sub-agent result" in result.output

    @pytest.mark.asyncio
    async def test_execute_no_task(self, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        tool = SpawnAgentTool(coordinator)
        result = await tool.execute({}, tool_context)
        assert result.is_error
        assert result.error is not None
        assert "non-empty" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_forwards_override_args(self, tool_context: ToolContext) -> None:
        coordinator = AsyncMock()
        coordinator.spawn = AsyncMock(return_value="ok")
        tool = SpawnAgentTool(coordinator)

        await tool.execute(
            {"task": "do X", "max_iterations": 10, "timeout": 60},
            tool_context,
        )
        # Coordinator receives the override values by keyword.
        coordinator.spawn.assert_awaited_once()
        call = coordinator.spawn.await_args
        assert call.args == ("do X",)
        assert call.kwargs == {"max_iterations": 10, "timeout": 60}


def test_default_constants() -> None:
    # Pin the defaults so they can't silently drift — these show up
    # in the tool schema description and the docs.
    assert DEFAULT_MAX_ITERATIONS == 25
    assert DEFAULT_TIMEOUT_SECONDS == 300
