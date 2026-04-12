"""Tests for extended thinking support (Unit 1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.llm.client import ChatResponse, LLMClient

# ---------------------------------------------------------------------------
# Config: thinking_budget field
# ---------------------------------------------------------------------------


def test_thinking_budget_defaults_zero():
    """thinking_budget defaults to 0 (disabled)."""
    from godspeed.config import GodspeedSettings

    with patch("godspeed.config.DEFAULT_GLOBAL_DIR", MagicMock(exists=lambda: False)):
        settings = GodspeedSettings(model="test")
    assert settings.thinking_budget == 0


def test_thinking_budget_config_value():
    """thinking_budget can be set via constructor."""
    from godspeed.config import GodspeedSettings

    with patch("godspeed.config.DEFAULT_GLOBAL_DIR", MagicMock(exists=lambda: False)):
        settings = GodspeedSettings(model="test", thinking_budget=10000)
    assert settings.thinking_budget == 10000


# ---------------------------------------------------------------------------
# LLMClient: thinking parameter passed to Anthropic models
# ---------------------------------------------------------------------------


def test_llm_client_stores_thinking_budget():
    """LLMClient stores thinking_budget from constructor."""
    client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=8000)
    assert client.thinking_budget == 8000


def test_is_anthropic_model_true():
    """Claude models are correctly identified as Anthropic."""
    client = LLMClient(model="claude-sonnet-4-20250514")
    assert client._is_anthropic_model() is True
    assert client._is_anthropic_model("anthropic/claude-3.5-sonnet") is True


def test_is_anthropic_model_false():
    """Non-Claude models are not Anthropic."""
    client = LLMClient(model="gpt-4o")
    assert client._is_anthropic_model() is False
    assert client._is_anthropic_model("ollama/qwen3:4b") is False


@pytest.mark.asyncio
async def test_thinking_param_added_for_claude():
    """When thinking_budget > 0 and model is Claude, thinking param is added."""
    client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=10000)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello",
                tool_calls=None,
                thinking=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 10000


@pytest.mark.asyncio
async def test_thinking_param_skipped_for_non_claude():
    """When model is not Claude, thinking param is not added."""
    client = LLMClient(model="gpt-4o", thinking_budget=10000)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello",
                tool_calls=None,
                thinking=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("gpt-4o", [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "thinking" not in call_kwargs


@pytest.mark.asyncio
async def test_thinking_param_skipped_when_zero():
    """When thinking_budget is 0, no thinking param even for Claude."""
    client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=0)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello",
                tool_calls=None,
                thinking=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "thinking" not in call_kwargs


# ---------------------------------------------------------------------------
# ChatResponse: thinking field
# ---------------------------------------------------------------------------


def test_chat_response_thinking_field():
    """ChatResponse includes thinking field."""
    resp = ChatResponse(content="hello", thinking="I need to think about this...")
    assert resp.thinking == "I need to think about this..."


def test_chat_response_thinking_default_empty():
    """ChatResponse thinking defaults to empty string."""
    resp = ChatResponse(content="hello")
    assert resp.thinking == ""


# ---------------------------------------------------------------------------
# Agent loop: on_thinking callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_thinking_callback_called(tmp_path):
    """Agent loop calls on_thinking when response has thinking content."""
    from godspeed.agent.conversation import Conversation
    from godspeed.tools.base import ToolContext
    from godspeed.tools.registry import ToolRegistry

    # Mock LLM to return a response with thinking
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(
        return_value=ChatResponse(
            content="The answer is 42.",
            thinking="Let me reason step by step...",
            finish_reason="stop",
        )
    )
    mock_llm.stream_chat = AsyncMock()

    conversation = Conversation(system_prompt="test", model="test", max_tokens=10000)
    registry = ToolRegistry()
    context = ToolContext(cwd=tmp_path, session_id="test")

    thinking_texts: list[str] = []

    def capture_thinking(text: str) -> None:
        thinking_texts.append(text)

    from godspeed.agent.loop import agent_loop

    await agent_loop(
        user_input="What is the meaning of life?",
        conversation=conversation,
        llm_client=mock_llm,
        tool_registry=registry,
        tool_context=context,
        on_thinking=capture_thinking,
    )

    assert len(thinking_texts) == 1
    assert "step by step" in thinking_texts[0]


@pytest.mark.asyncio
async def test_on_thinking_not_called_when_empty(tmp_path):
    """Agent loop does not call on_thinking when thinking is empty."""
    from godspeed.agent.conversation import Conversation
    from godspeed.tools.base import ToolContext
    from godspeed.tools.registry import ToolRegistry

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(
        return_value=ChatResponse(
            content="Hello!",
            thinking="",
            finish_reason="stop",
        )
    )

    conversation = Conversation(system_prompt="test", model="test", max_tokens=10000)
    registry = ToolRegistry()
    context = ToolContext(cwd=tmp_path, session_id="test")

    thinking_texts: list[str] = []

    from godspeed.agent.loop import agent_loop

    await agent_loop(
        user_input="Hi",
        conversation=conversation,
        llm_client=mock_llm,
        tool_registry=registry,
        tool_context=context,
        on_thinking=lambda t: thinking_texts.append(t),
    )

    assert len(thinking_texts) == 0


# ---------------------------------------------------------------------------
# TUI: /think command
# ---------------------------------------------------------------------------


def test_think_command_toggle_on(tmp_path):
    """'/think' toggles thinking ON with default 10k budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think")
    assert result.handled
    assert llm_client.thinking_budget == 10_000


def test_think_command_toggle_off(tmp_path):
    """'/think' toggles thinking OFF when already on."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 10_000
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think")
    assert result.handled
    assert llm_client.thinking_budget == 0


def test_think_command_set_budget(tmp_path):
    """'/think 20000' sets a custom budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think 20000")
    assert result.handled
    assert llm_client.thinking_budget == 20_000


def test_think_command_reject_small_budget(tmp_path):
    """'/think 500' rejects budgets under 1000."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think 500")
    assert result.handled
    assert llm_client.thinking_budget == 0  # unchanged


def test_think_command_off_keyword(tmp_path):
    """'/think off' explicitly disables."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 10_000
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think off")
    assert result.handled
    assert llm_client.thinking_budget == 0


# ---------------------------------------------------------------------------
# TUI: format_thinking
# ---------------------------------------------------------------------------


def test_format_thinking_nonempty(capsys):
    """format_thinking displays non-empty text."""
    from godspeed.tui.output import format_thinking

    # Just verify it doesn't raise
    format_thinking("I'm thinking about this problem...")


def test_format_thinking_empty():
    """format_thinking does nothing for empty text."""
    from godspeed.tui.output import format_thinking

    format_thinking("")  # Should not raise
    format_thinking("   ")  # Whitespace only
