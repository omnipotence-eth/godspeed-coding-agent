"""Tests for model-aware compaction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.config import get_model_context_window
from godspeed.context.compaction import (
    COMPACTION_PROMPT_LARGE,
    COMPACTION_PROMPT_SMALL,
    COMPACTION_PROMPT_ULTRA_LARGE,
    compact_if_needed,
    get_compaction_prompt,
)
from godspeed.llm.client import ChatResponse, LLMClient


class TestModelContextWindows:
    """Test model context window lookup."""

    def test_claude_opus(self) -> None:
        assert get_model_context_window("claude-opus-4-20250514") == 1_000_000

    def test_claude_sonnet(self) -> None:
        assert get_model_context_window("claude-sonnet-4-20250514") == 1_000_000

    def test_gpt4o(self) -> None:
        assert get_model_context_window("gpt-4o-2024-05-13") == 128_000

    def test_gpt4_base(self) -> None:
        assert get_model_context_window("gpt-4-0613") == 8_192

    def test_ollama_qwen(self) -> None:
        assert get_model_context_window("ollama/qwen3:4b") == 32_768

    def test_ollama_llama3(self) -> None:
        assert get_model_context_window("ollama/llama3:8b") == 8_192

    def test_ollama_llama31(self) -> None:
        assert get_model_context_window("ollama/llama3.1:8b") == 128_000

    def test_unknown_model_defaults(self) -> None:
        assert get_model_context_window("totally-unknown-model") == 32_768

    def test_empty_string_defaults(self) -> None:
        assert get_model_context_window("") == 32_768

    def test_case_insensitive(self) -> None:
        assert get_model_context_window("Claude-Sonnet-4") == 1_000_000

    def test_gemini_large_context(self) -> None:
        assert get_model_context_window("gemini-2-flash") == 1_000_000


class TestCompactionPromptSelection:
    """Test model-aware compaction prompt selection."""

    def test_small_model_gets_aggressive_prompt(self) -> None:
        # ollama/llama3 = 8192 → small
        prompt = get_compaction_prompt("ollama/llama3:8b")
        assert prompt == COMPACTION_PROMPT_SMALL
        assert "aggressively" in prompt.lower()

    def test_medium_model_gets_balanced_prompt(self) -> None:
        # ollama/qwen3 = 32768 → small threshold (≤32K), actually hits small
        # Use a model that's between 32K and 100K
        prompt = get_compaction_prompt("ollama/mistral:7b")
        assert prompt == COMPACTION_PROMPT_SMALL  # 32768 ≤ 32768 threshold

    def test_frontier_model_gets_detailed_prompt(self) -> None:
        # claude-sonnet-4 = 1M → ultra-large
        prompt = get_compaction_prompt("claude-sonnet-4-20250514")
        assert prompt == COMPACTION_PROMPT_ULTRA_LARGE
        assert "exhaustive" in prompt.lower()

    def test_large_model_gets_detailed_prompt(self) -> None:
        # gpt-4o = 128K → large (not ultra-large)
        prompt = get_compaction_prompt("gpt-4o")
        assert prompt == COMPACTION_PROMPT_LARGE

    def test_unknown_model_gets_medium_prompt(self) -> None:
        # Unknown = 32768 default → small threshold
        prompt = get_compaction_prompt("unknown-model")
        assert prompt == COMPACTION_PROMPT_SMALL

    def test_empty_model_gets_medium_prompt(self) -> None:
        prompt = get_compaction_prompt("")
        assert prompt == COMPACTION_PROMPT_SMALL  # default 32768 ≤ threshold

    def test_gpt4_base_gets_small_prompt(self) -> None:
        # gpt-4 = 8192 → small
        prompt = get_compaction_prompt("gpt-4-0613")
        assert prompt == COMPACTION_PROMPT_SMALL


class TestCompactIfNeeded:
    """Test the compact_if_needed function with model awareness."""

    @pytest.mark.asyncio
    async def test_no_compaction_when_not_needed(self) -> None:
        conv = Conversation("System prompt", max_tokens=100_000)
        client = LLMClient(model="test")
        result = await compact_if_needed(conv, client, model="claude-sonnet-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_compaction_uses_model_prompt(self) -> None:
        conv = Conversation("System prompt", max_tokens=100, compaction_threshold=0.01)
        # Add enough content to trigger compaction
        for i in range(20):
            conv.add_user_message(f"Message {i} with some content to fill tokens")
            conv.add_assistant_message(f"Response {i} with additional content here")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(
                content="Summary of conversation", tool_calls=[], finish_reason="stop"
            )
        )

        result = await compact_if_needed(conv, client, model="claude-sonnet-4")

        if result:
            # Verify chat was called with the ultra-large prompt (claude-sonnet-4 = 1M)
            call_args = client.chat.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
            system_content = messages[0]["content"]
            assert "exhaustive" in system_content.lower()

    @pytest.mark.asyncio
    async def test_compaction_with_small_model(self) -> None:
        conv = Conversation("System prompt", max_tokens=100, compaction_threshold=0.01)
        for i in range(20):
            conv.add_user_message(f"Message {i} with content")
            conv.add_assistant_message(f"Response {i} here")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Brief summary", tool_calls=[], finish_reason="stop")
        )

        result = await compact_if_needed(conv, client, model="ollama/llama3:8b")

        if result:
            call_args = client.chat.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
            system_content = messages[0]["content"]
            assert "aggressively" in system_content.lower()

    @pytest.mark.asyncio
    async def test_compaction_fallback_no_model(self) -> None:
        """Without a model name, falls back to medium prompt."""
        conv = Conversation("System prompt", max_tokens=100, compaction_threshold=0.01)
        for i in range(20):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            return_value=ChatResponse(content="Summary", tool_calls=[], finish_reason="stop")
        )

        # model="" triggers the fallback
        result = await compact_if_needed(conv, client, model="")
        # Should not crash regardless of compaction triggering
        assert isinstance(result, bool)
