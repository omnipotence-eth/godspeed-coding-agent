"""Tests for the agent result/metrics contract (v2.5.0)."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import agent_loop
from godspeed.agent.result import (
    EXIT_REASON_TO_CODE,
    AgentMetrics,
    ExitCode,
    ExitReason,
    ToolCallRecord,
)
from godspeed.llm.client import BudgetExceededError, ChatResponse, LLMClient
from godspeed.tools.base import ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _text_resp(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _tool_resp(name: str, args: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[{"id": "call_1", "function": {"name": name, "arguments": json.dumps(args)}}],
        finish_reason="tool_calls",
    )


class TestExitCode:
    """Exit code contract — values are stable and documented."""

    def test_success_is_zero(self) -> None:
        assert ExitCode.SUCCESS == 0

    def test_interrupted_is_130_posix_convention(self) -> None:
        """SIGINT exit code by POSIX convention is 128 + 2."""
        assert ExitCode.INTERRUPTED == 130

    def test_each_reason_maps_to_code(self) -> None:
        """Every ExitReason has a unique ExitCode — no reason unmapped."""
        for reason in ExitReason:
            assert reason in EXIT_REASON_TO_CODE

    def test_no_duplicate_codes(self) -> None:
        """No two reasons share an exit code."""
        codes = list(EXIT_REASON_TO_CODE.values())
        assert len(codes) == len(set(codes))


class TestAgentMetrics:
    """AgentMetrics accumulator behavior."""

    def test_defaults(self) -> None:
        m = AgentMetrics()
        assert m.iterations_used == 0
        assert m.tool_call_count == 0
        assert m.tool_error_count == 0
        assert m.must_fix_injections == 0
        assert m.end_time is None
        assert m.exit_reason == ExitReason.STOPPED

    def test_record_must_fix_injection_increments(self) -> None:
        m = AgentMetrics()
        m.record_must_fix_injection()
        m.record_must_fix_injection()
        assert m.must_fix_injections == 2

    def test_record_tool_call(self) -> None:
        m = AgentMetrics()
        m.record_tool_call("shell", is_error=False)
        m.record_tool_call("file_edit", is_error=True)
        assert m.tool_call_count == 2
        assert m.tool_error_count == 1
        assert m.tool_calls[0] == ToolCallRecord(name="shell", is_error=False)

    def test_finalize_sets_reason_and_end_time(self) -> None:
        m = AgentMetrics()
        m.finalize(ExitReason.BUDGET_EXCEEDED)
        assert m.exit_reason == ExitReason.BUDGET_EXCEEDED
        assert m.end_time is not None
        assert m.duration_seconds >= 0

    def test_exit_code_derived_from_reason(self) -> None:
        m = AgentMetrics()
        m.finalize(ExitReason.MAX_ITERATIONS)
        assert m.exit_code == ExitCode.MAX_ITERATIONS

    def test_duration_before_finalize(self) -> None:
        """duration_seconds should return a non-negative value (doesn't raise)
        even before finalize() has set end_time. Windows monotonic() has
        coarse granularity so we only assert non-negative, not >0."""
        m = AgentMetrics()
        time.sleep(0.01)
        d = m.duration_seconds
        assert d >= 0.0


class TestAgentLoopPopulatesMetrics:
    """The agent loop must populate metrics at every exit path when the
    metrics arg is provided. This is what the headless runner relies on
    to report accurate exit codes in CI.
    """

    @pytest.mark.asyncio
    async def test_success_populates_stopped(self, tool_context) -> None:
        metrics = AgentMetrics()
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_text_resp("all done"))

        result = await agent_loop(
            "hi", conversation, client, registry, tool_context, metrics=metrics
        )

        assert result == "all done"
        assert metrics.exit_reason == ExitReason.STOPPED
        assert metrics.exit_code == ExitCode.SUCCESS
        assert metrics.iterations_used == 1
        assert metrics.end_time is not None

    @pytest.mark.asyncio
    async def test_budget_exceeded_populates_budget(self, tool_context) -> None:
        metrics = AgentMetrics()
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=BudgetExceededError(spent=5.0, limit=1.0))

        result = await agent_loop(
            "hi", conversation, client, registry, tool_context, metrics=metrics
        )

        assert "Budget exceeded" in result
        assert metrics.exit_reason == ExitReason.BUDGET_EXCEEDED
        assert metrics.exit_code == ExitCode.BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_llm_error_populates_llm_error(self, tool_context) -> None:
        metrics = AgentMetrics()
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("provider 500"))

        result = await agent_loop(
            "hi", conversation, client, registry, tool_context, metrics=metrics
        )

        assert result.startswith("Error: LLM call failed")
        assert metrics.exit_reason == ExitReason.LLM_ERROR
        assert metrics.exit_code == ExitCode.LLM_ERROR

    @pytest.mark.asyncio
    async def test_max_iterations_populates_max_iter(self, tool_context) -> None:
        """Looping forever on tool calls should exit at MAX_ITERATIONS."""
        metrics = AgentMetrics()
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell"))
        client = LLMClient(model="test")
        # Always return a tool call — never a text response.
        client.chat = AsyncMock(return_value=_tool_resp("shell", {"command": "ls"}))

        result = await agent_loop(
            "loop forever",
            conversation,
            client,
            registry,
            tool_context,
            metrics=metrics,
            max_iterations=3,
        )

        assert "maximum iterations" in result.lower()
        assert metrics.exit_reason == ExitReason.MAX_ITERATIONS
        assert metrics.exit_code == ExitCode.MAX_ITERATIONS
        assert metrics.iterations_used == 3

    @pytest.mark.asyncio
    async def test_tool_calls_are_recorded(self, tool_context) -> None:
        """Every tool dispatch populates metrics.tool_calls."""
        metrics = AgentMetrics()
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _tool_resp("shell", {"command": "ls"}),
                _text_resp("done"),
            ]
        )

        await agent_loop("run ls", conversation, client, registry, tool_context, metrics=metrics)

        assert metrics.tool_call_count == 1
        assert metrics.tool_calls[0].name == "shell"
        assert metrics.tool_calls[0].is_error is False

    @pytest.mark.asyncio
    async def test_metrics_arg_is_optional(self, tool_context) -> None:
        """Callers that don't pass metrics must continue to work (backwards-compat)."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_text_resp("fine"))

        result = await agent_loop("hi", conversation, client, registry, tool_context)
        assert result == "fine"
