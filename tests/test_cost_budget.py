"""Tests for cost budget enforcement (Unit 5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.llm.client import BudgetExceededError, LLMClient

# ---------------------------------------------------------------------------
# Config: max_cost_usd field
# ---------------------------------------------------------------------------


def test_max_cost_defaults_zero():
    """max_cost_usd defaults to 0 (unlimited)."""
    from godspeed.config import GodspeedSettings

    with patch("godspeed.config.DEFAULT_GLOBAL_DIR", MagicMock(exists=lambda: False)):
        settings = GodspeedSettings(model="test")
    assert settings.max_cost_usd == 0.0


def test_max_cost_config_value():
    """max_cost_usd can be set via constructor."""
    from godspeed.config import GodspeedSettings

    with patch("godspeed.config.DEFAULT_GLOBAL_DIR", MagicMock(exists=lambda: False)):
        settings = GodspeedSettings(model="test", max_cost_usd=5.0)
    assert settings.max_cost_usd == 5.0


# ---------------------------------------------------------------------------
# LLMClient: budget tracking
# ---------------------------------------------------------------------------


def test_llm_client_stores_max_cost():
    """LLMClient stores max_cost_usd."""
    client = LLMClient(model="gpt-4o", max_cost_usd=10.0)
    assert client.max_cost_usd == 10.0
    assert client.total_cost_usd == 0.0


def test_check_budget_no_limit():
    """_check_budget does nothing when limit is 0 (unlimited)."""
    client = LLMClient(model="gpt-4o", max_cost_usd=0.0)
    client.total_cost_usd = 100.0
    client._check_budget()  # Should not raise


def test_check_budget_under_limit():
    """_check_budget does nothing when under limit."""
    client = LLMClient(model="gpt-4o", max_cost_usd=10.0)
    client.total_cost_usd = 5.0
    client._check_budget()  # Should not raise


def test_check_budget_exceeded():
    """_check_budget raises BudgetExceededError when over limit."""
    client = LLMClient(model="gpt-4o", max_cost_usd=5.0)
    client.total_cost_usd = 5.01
    with pytest.raises(BudgetExceededError) as exc_info:
        client._check_budget()
    assert exc_info.value.spent == 5.01
    assert exc_info.value.limit == 5.0


def test_budget_exceeded_error_message():
    """BudgetExceededError has informative message."""
    err = BudgetExceededError(spent=7.50, limit=5.00)
    assert "$7.50" in str(err)
    assert "$5.00" in str(err)


@pytest.mark.asyncio
async def test_cost_tracked_after_call():
    """LLM call tracks cost in total_cost_usd."""
    client = LLMClient(model="claude-sonnet-4-20250514")

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Hello", tool_calls=None, thinking=None),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=1000, completion_tokens=500)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], None)

    # Claude Sonnet pricing: $3/M input + $15/M output
    # 1000 input tokens = $0.003, 500 output tokens = $0.0075
    assert client.total_cost_usd > 0
    assert client.total_cost_usd == pytest.approx(0.0105, abs=0.001)


@pytest.mark.asyncio
async def test_budget_exceeded_during_call():
    """LLM call raises BudgetExceededError when budget exceeded mid-session."""
    client = LLMClient(model="claude-sonnet-4-20250514", max_cost_usd=0.001)
    # Pre-fill some cost close to limit
    client.total_cost_usd = 0.0009

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Hello", tool_calls=None, thinking=None),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=1000, completion_tokens=500)

    with (
        patch("godspeed.llm.client._get_litellm") as mock_litellm,
        pytest.raises(BudgetExceededError),
    ):
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], None)


@pytest.mark.asyncio
async def test_ollama_always_free():
    """Ollama models never accumulate cost."""
    client = LLMClient(model="ollama/qwen3:4b", max_cost_usd=0.001)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Hello", tool_calls=None, thinking=None),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10000, completion_tokens=5000)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("ollama_chat/qwen3:4b", [{"role": "user", "content": "hi"}], None)

    assert client.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Agent loop: budget exceeded handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_catches_budget_exceeded(tmp_path):
    """Agent loop catches BudgetExceededError and returns informative message."""
    from godspeed.agent.conversation import Conversation
    from godspeed.agent.loop import agent_loop
    from godspeed.tools.base import ToolContext
    from godspeed.tools.registry import ToolRegistry

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(side_effect=BudgetExceededError(spent=5.50, limit=5.00))

    conversation = Conversation(system_prompt="test", model="test", max_tokens=10000)
    registry = ToolRegistry()
    context = ToolContext(cwd=tmp_path, session_id="test")

    result = await agent_loop(
        user_input="Write some code",
        conversation=conversation,
        llm_client=mock_llm,
        tool_registry=registry,
        tool_context=context,
    )

    assert "Budget exceeded" in result
    assert "$5.50" in result or "5.5" in result
    assert "/budget" in result


# ---------------------------------------------------------------------------
# TUI: /budget command
# ---------------------------------------------------------------------------


def test_budget_command_show(tmp_path):
    """'/budget' with no args shows current cost and limit."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.total_cost_usd = 1.50
    llm_client.max_cost_usd = 10.0
    llm_client.model = "claude-sonnet-4-20250514"
    llm_client.total_input_tokens = 5000
    llm_client.total_output_tokens = 2000

    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/budget")
    assert result.handled


def test_budget_command_set(tmp_path):
    """'/budget 5.00' sets the budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.max_cost_usd = 0.0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/budget 5.00")
    assert result.handled
    assert llm_client.max_cost_usd == 5.0


def test_budget_command_set_with_dollar(tmp_path):
    """'/budget $10' strips $ prefix."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.max_cost_usd = 0.0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/budget $10")
    assert result.handled
    assert llm_client.max_cost_usd == 10.0


def test_budget_command_off(tmp_path):
    """'/budget off' disables budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.max_cost_usd = 5.0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/budget off")
    assert result.handled
    assert llm_client.max_cost_usd == 0.0


def test_budget_command_unlimited(tmp_path):
    """'/budget unlimited' disables budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.max_cost_usd = 5.0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/budget unlimited")
    assert result.handled
    assert llm_client.max_cost_usd == 0.0


def test_budget_command_invalid(tmp_path):
    """'/budget abc' shows error."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.max_cost_usd = 0.0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/budget abc")
    assert result.handled
    assert llm_client.max_cost_usd == 0.0  # unchanged
