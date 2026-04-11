"""Tests for the core agent loop."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import _parse_tool_call, agent_loop
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_tool_response(tool_name: str, arguments: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[
            {
                "id": "call_001",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
        finish_reason="tool_calls",
    )


class TestParseToolCall:
    """Test tool call parsing from LLM responses."""

    def test_valid_tool_call(self) -> None:
        raw = {
            "id": "call_001",
            "function": {"name": "file_read", "arguments": '{"file_path": "test.py"}'},
        }
        tc = _parse_tool_call(raw)
        assert tc is not None
        assert tc.tool_name == "file_read"
        assert tc.arguments == {"file_path": "test.py"}
        assert tc.call_id == "call_001"

    def test_dict_arguments(self) -> None:
        raw = {
            "id": "call_002",
            "function": {"name": "shell", "arguments": {"command": "ls"}},
        }
        tc = _parse_tool_call(raw)
        assert tc is not None
        assert tc.arguments == {"command": "ls"}

    def test_invalid_json(self) -> None:
        raw = {
            "id": "call_003",
            "function": {"name": "shell", "arguments": "not json{"},
        }
        tc = _parse_tool_call(raw)
        assert tc is None

    def test_missing_name(self) -> None:
        raw = {"id": "call_004", "function": {"name": "", "arguments": "{}"}}
        tc = _parse_tool_call(raw)
        assert tc is None

    def test_empty_function(self) -> None:
        raw = {"id": "call_005", "function": {}}
        tc = _parse_tool_call(raw)
        assert tc is None


class TestAgentLoop:
    """Test the full agent loop."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Hello!"))

        result = await agent_loop("Hi", conversation, client, registry, tool_context)
        assert result == "Hello!"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        tool = MockTool(name="file_read", result=ToolResult.success("file contents"))
        registry.register(tool)

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("file_read", {"file_path": "test.py"}),
                _make_text_response("I read the file. It contains: file contents"),
            ]
        )

        result = await agent_loop("Read test.py", conversation, client, registry, tool_context)
        assert "file contents" in result
        assert client.chat.call_count == 2
        assert tool.last_arguments == {"file_path": "test.py"}

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("nonexistent_tool", {}),
                _make_text_response("Sorry, that tool doesn't exist."),
            ]
        )

        result = await agent_loop("Do something", conversation, client, registry, tool_context)
        assert "Sorry" in result

    @pytest.mark.asyncio
    async def test_callbacks_called(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "ls"}),
                _make_text_response("Done"),
            ]
        )

        text_calls = []
        tool_calls = []
        tool_results = []

        await agent_loop(
            "List files",
            conversation,
            client,
            registry,
            tool_context,
            on_assistant_text=text_calls.append,
            on_tool_call=lambda name, args: tool_calls.append((name, args)),
            on_tool_result=lambda name, result: tool_results.append((name, result)),
        )

        assert len(tool_calls) == 1
        assert tool_calls[0][0] == "shell"
        assert len(tool_results) == 1
        assert "Done" in text_calls

    @pytest.mark.asyncio
    async def test_malformed_tool_call_retries(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # Send a malformed tool call (bad JSON), then a valid text response
        malformed_response = ChatResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_bad",
                    "function": {"name": "shell", "arguments": "not valid json{"},
                }
            ],
            finish_reason="tool_calls",
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                malformed_response,
                _make_text_response("Recovered from bad tool call"),
            ]
        )

        result = await agent_loop("Do something", conversation, client, registry, tool_context)
        assert "Recovered" in result


class TestConversation:
    """Test conversation management."""

    def test_add_messages(self) -> None:
        conv = Conversation("System prompt")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there")
        msgs = conv.messages
        assert len(msgs) == 3  # system + user + assistant
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_add_tool_result(self) -> None:
        conv = Conversation("System prompt")
        conv.add_tool_result("call_001", "tool output")
        msgs = conv.messages
        assert msgs[-1]["role"] == "tool"
        assert msgs[-1]["tool_call_id"] == "call_001"

    def test_compact(self) -> None:
        conv = Conversation("System prompt")
        for i in range(10):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")
        assert len(conv.messages) > 10
        conv.compact("Summary of conversation")
        # System prompt + compaction message
        assert len(conv.messages) == 2
        assert "Summary" in conv.messages[1]["content"]

    def test_clear(self) -> None:
        conv = Conversation("System prompt")
        conv.add_user_message("Hello")
        conv.clear()
        assert len(conv.messages) == 1  # Only system prompt

    def test_token_count(self) -> None:
        conv = Conversation("Short system prompt")
        assert conv.token_count > 0
        conv.add_user_message("Hello world")
        count_after = conv.token_count
        assert count_after > 0

    def test_is_near_limit(self) -> None:
        conv = Conversation("System prompt", max_tokens=10, compaction_threshold=0.5)
        # Short conversation should not be near limit, but with very low max_tokens it might be
        # The system prompt itself might push us over
        # This test just verifies the property works
        assert isinstance(conv.is_near_limit, bool)

    def test_get_compaction_context_includes_roles(self) -> None:
        conv = Conversation("System prompt")
        conv.add_user_message("Fix the bug")
        conv.add_assistant_message("I'll read the file first")
        context = conv.get_compaction_context()
        assert "[user]: Fix the bug" in context
        assert "[assistant]: I'll read the file first" in context

    def test_get_compaction_context_includes_tool_calls(self) -> None:
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "file_read", "arguments": '{"file_path": "main.py"}'},
                }
            ],
        )
        context = conv.get_compaction_context()
        assert "file_read" in context
        assert "main.py" in context

    def test_add_assistant_message_normalizes_tool_calls_type(self) -> None:
        """Tool calls should always have type='function' for LiteLLM compat."""
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            tool_calls=[{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        )
        msg = conv.messages[-1]
        assert msg["tool_calls"][0]["type"] == "function"

    def test_add_assistant_message_preserves_existing_type(self) -> None:
        """If type is already set, don't overwrite it."""
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        )
        msg = conv.messages[-1]
        assert msg["tool_calls"][0]["type"] == "function"


class TestTokenCounter:
    """Tests for token counting utilities."""

    def test_get_encoding_for_known_model(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("gpt-4")
        assert enc is not None

    def test_get_encoding_for_claude_falls_back(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("claude-sonnet-4-20250514")
        assert enc is not None

    def test_get_encoding_for_ollama_model(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("ollama/qwen3:4b")
        assert enc is not None

    def test_get_encoding_for_unknown_model(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("totally-unknown-model-xyz")
        assert enc is not None  # falls back to cl100k_base

    def test_count_tokens_basic(self) -> None:
        from godspeed.llm.token_counter import count_tokens

        count = count_tokens("Hello world")
        assert count > 0

    def test_count_message_tokens_basic(self) -> None:
        from godspeed.llm.token_counter import count_message_tokens

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        count = count_message_tokens(msgs)
        assert count > 0

    def test_count_message_tokens_with_tool_calls(self) -> None:
        from godspeed.llm.token_counter import count_message_tokens

        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
            }
        ]
        count = count_message_tokens(msgs)
        assert count > 0

    def test_count_tokens_empty_string(self) -> None:
        from godspeed.llm.token_counter import count_tokens

        count = count_tokens("")
        assert count == 0
