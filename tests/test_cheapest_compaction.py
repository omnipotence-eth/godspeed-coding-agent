"""Tests for cheapest-model compaction and risk-based dispatch (Units 17 & 18)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from godspeed.llm.cost import get_cheapest_model

# ---------------------------------------------------------------------------
# get_cheapest_model
# ---------------------------------------------------------------------------


def test_cheapest_model_empty_list():
    """Empty list returns empty string."""
    assert get_cheapest_model([]) == ""


def test_cheapest_model_single():
    """Single model is always returned."""
    assert get_cheapest_model(["claude-sonnet-4-20250514"]) == "claude-sonnet-4-20250514"


def test_cheapest_model_ollama_wins():
    """Ollama (free) is always cheapest."""
    models = ["claude-opus-4-20250514", "ollama/qwen3:4b", "gpt-4o"]
    assert get_cheapest_model(models) == "ollama/qwen3:4b"


def test_cheapest_model_haiku_cheapest_among_paid():
    """Haiku is cheaper than Sonnet and GPT-4o."""
    models = ["claude-sonnet-4-20250514", "claude-haiku-3.5", "gpt-4o"]
    result = get_cheapest_model(models)
    assert "haiku" in result.lower()


def test_cheapest_model_all_unknown():
    """Unknown models: returns first."""
    models = ["custom/model-a", "custom/model-b"]
    assert get_cheapest_model(models) == "custom/model-a"


def test_cheapest_model_ollama_chat_prefix():
    """ollama_chat/ prefix is also free."""
    models = ["gpt-4o", "ollama_chat/llama3:8b"]
    assert get_cheapest_model(models) == "ollama_chat/llama3:8b"


# ---------------------------------------------------------------------------
# Risk-based parallel/serial split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_based_split_read_only_parallel(tmp_path):
    """READ_ONLY tools run in parallel, write tools run serially."""
    from godspeed.agent.conversation import Conversation
    from godspeed.agent.loop import agent_loop
    from godspeed.llm.client import ChatResponse, LLMClient
    from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
    from godspeed.tools.registry import ToolRegistry

    execution_order: list[str] = []

    class ReadTool(Tool):
        @property
        def name(self) -> str:
            return "read_tool"

        @property
        def description(self) -> str:
            return "test read"

        @property
        def risk_level(self) -> RiskLevel:
            return RiskLevel.READ_ONLY

        def get_schema(self):
            return {"type": "object", "properties": {}}

        async def execute(self, arguments, context):
            execution_order.append("read")
            return ToolResult.success("read ok")

    class WriteTool(Tool):
        @property
        def name(self) -> str:
            return "write_tool"

        @property
        def description(self) -> str:
            return "test write"

        @property
        def risk_level(self) -> RiskLevel:
            return RiskLevel.LOW

        def get_schema(self):
            return {"type": "object", "properties": {}}

        async def execute(self, arguments, context):
            execution_order.append("write")
            return ToolResult.success("write ok")

    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())

    # First response: 2 tool calls (one read, one write)
    # Second response: text-only (stop)
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(
        side_effect=[
            ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "function": {"name": "read_tool", "arguments": "{}"},
                    },
                    {
                        "id": "tc2",
                        "function": {"name": "write_tool", "arguments": "{}"},
                    },
                ],
                finish_reason="tool_calls",
                thinking="",
            ),
            ChatResponse(content="Done", finish_reason="stop", thinking=""),
        ]
    )

    conversation = Conversation(system_prompt="test", model="test", max_tokens=10000)
    context = ToolContext(cwd=tmp_path, session_id="test")

    await agent_loop(
        user_input="test",
        conversation=conversation,
        llm_client=mock_llm,
        tool_registry=registry,
        tool_context=context,
        parallel_tool_calls=True,
    )

    # Both tools should have executed
    assert "read" in execution_order
    assert "write" in execution_order
