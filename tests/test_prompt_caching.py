"""Tests for prompt caching support."""

from __future__ import annotations

from godspeed.llm.client import LLMClient


class TestPromptCaching:
    """Test _apply_prompt_caching static method."""

    def test_adds_cache_control_for_claude(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        # System message should be converted to content block with cache_control
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # User message should be unchanged
        assert result[1]["content"] == "Hello"

    def test_adds_cache_control_for_gpt4o(self) -> None:
        messages = [{"role": "system", "content": "System prompt"}]
        result = LLMClient._apply_prompt_caching("gpt-4o", messages)
        assert isinstance(result[0]["content"], list)
        assert "cache_control" in result[0]["content"][0]

    def test_no_cache_for_ollama(self) -> None:
        messages = [{"role": "system", "content": "System prompt"}]
        result = LLMClient._apply_prompt_caching("ollama/qwen3:4b", messages)
        # Should be unchanged
        assert result[0]["content"] == "System prompt"

    def test_no_cache_for_unknown_model(self) -> None:
        messages = [{"role": "system", "content": "System prompt"}]
        result = LLMClient._apply_prompt_caching("some-random-model", messages)
        assert result[0]["content"] == "System prompt"

    def test_preserves_non_system_messages(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User input"},
            {"role": "assistant", "content": "Response"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        assert result[1] == messages[1]
        assert result[2] == messages[2]

    def test_handles_already_structured_content(self) -> None:
        """System message with list content should not be double-wrapped."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Already structured"}]},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", messages)
        # Should pass through unchanged since content is already a list
        assert result[0]["content"] == [{"type": "text", "text": "Already structured"}]

    def test_empty_messages(self) -> None:
        result = LLMClient._apply_prompt_caching("claude-sonnet-4-20250514", [])
        assert result == []
