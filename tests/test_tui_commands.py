"""Tests for TUI slash commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import RiskLevel
from godspeed.tui.commands import Commands


@pytest.fixture
def conversation() -> Conversation:
    return Conversation("You are a coding agent.", max_tokens=100_000)


@pytest.fixture
def commands(conversation: Conversation, tmp_path: Path) -> Commands:
    llm_client = MagicMock()
    llm_client.model = "test-model"
    llm_client.fallback_models = []
    llm_client.total_input_tokens = 0
    llm_client.total_output_tokens = 0
    return Commands(
        conversation=conversation,
        llm_client=llm_client,
        permission_engine=None,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
    )


class TestExtendCommand:
    """Test /extend command for setting max iterations."""

    def test_extend_sets_value(self, commands: Commands) -> None:
        result = commands.dispatch("/extend 100")
        assert result is not None
        assert result.handled
        assert commands.max_iterations == 100

    def test_extend_no_args_shows_current(self, commands: Commands) -> None:
        result = commands.dispatch("/extend")
        assert result is not None
        assert result.handled
        # Default should be None (uses MAX_ITERATIONS)
        assert commands.max_iterations is None

    def test_extend_shows_updated_value(self, commands: Commands) -> None:
        commands.dispatch("/extend 75")
        assert commands.max_iterations == 75

    def test_extend_invalid_arg(self, commands: Commands) -> None:
        result = commands.dispatch("/extend abc")
        assert result is not None
        assert result.handled
        assert commands.max_iterations is None  # Unchanged

    def test_extend_zero_rejected(self, commands: Commands) -> None:
        result = commands.dispatch("/extend 0")
        assert result is not None
        assert result.handled
        assert commands.max_iterations is None  # Unchanged

    def test_extend_negative_rejected(self, commands: Commands) -> None:
        result = commands.dispatch("/extend -5")
        assert result is not None
        assert result.handled
        assert commands.max_iterations is None  # Unchanged

    def test_extend_minimum_one(self, commands: Commands) -> None:
        result = commands.dispatch("/extend 1")
        assert result is not None
        assert commands.max_iterations == 1


class TestContextCommand:
    """Test /context command for context window usage display."""

    def test_context_shows_usage(self, commands: Commands, conversation: Conversation) -> None:
        result = commands.dispatch("/context")
        assert result is not None
        assert result.handled

    def test_context_after_messages(self, commands: Commands, conversation: Conversation) -> None:
        conversation.add_user_message("Hello world")
        conversation.add_assistant_message("Hi there")
        result = commands.dispatch("/context")
        assert result is not None
        assert result.handled


class TestCommandDispatch:
    """Test command dispatch basics."""

    def test_unknown_command(self, commands: Commands) -> None:
        result = commands.dispatch("/unknown")
        assert result is not None
        assert result.handled

    def test_non_command_returns_none(self, commands: Commands) -> None:
        result = commands.dispatch("hello world")
        assert result is None

    def test_help_command(self, commands: Commands) -> None:
        result = commands.dispatch("/help")
        assert result is not None
        assert result.handled

    def test_quit_command(self, commands: Commands) -> None:
        result = commands.dispatch("/quit")
        assert result is not None
        assert result.should_quit

    def test_clear_command(self, commands: Commands, conversation: Conversation) -> None:
        conversation.add_user_message("test")
        result = commands.dispatch("/clear")
        assert result is not None
        assert result.handled
        # Only system prompt should remain
        assert len(conversation.messages) == 1


class TestPlanCommand:
    """Test /plan command for toggling plan mode."""

    @pytest.fixture
    def commands_with_perms(self, conversation: Conversation, tmp_path: Path) -> Commands:
        llm_client = MagicMock()
        llm_client.model = "test-model"
        llm_client.fallback_models = []
        llm_client.total_input_tokens = 0
        llm_client.total_output_tokens = 0
        engine = PermissionEngine(
            tool_risk_levels={"file_read": RiskLevel.READ_ONLY, "shell": RiskLevel.HIGH},
        )
        return Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=engine,
            audit_trail=None,
            session_id="test-session",
            cwd=tmp_path,
        )

    def test_plan_toggles_on(self, commands_with_perms: Commands) -> None:
        result = commands_with_perms.dispatch("/plan")
        assert result is not None
        assert result.handled
        assert commands_with_perms._permission_engine.plan_mode is True

    def test_plan_toggles_off(self, commands_with_perms: Commands) -> None:
        commands_with_perms.dispatch("/plan")  # on
        commands_with_perms.dispatch("/plan")  # off
        assert commands_with_perms._permission_engine.plan_mode is False

    def test_plan_without_permission_engine(self, commands: Commands) -> None:
        """Commands with no permission engine should handle /plan gracefully."""
        result = commands.dispatch("/plan")
        assert result is not None
        assert result.handled


class TestPauseResumeCommands:
    """Test /pause, /resume, and /guidance commands."""

    @pytest.fixture
    def commands_with_pause(self, conversation: Conversation, tmp_path: Path) -> Commands:
        import asyncio

        llm_client = MagicMock()
        llm_client.model = "test-model"
        llm_client.fallback_models = []
        llm_client.total_input_tokens = 0
        llm_client.total_output_tokens = 0
        pause_event = asyncio.Event()
        pause_event.set()  # Start running
        return Commands(
            conversation=conversation,
            llm_client=llm_client,
            permission_engine=None,
            audit_trail=None,
            session_id="test-session",
            cwd=tmp_path,
            pause_event=pause_event,
        )

    def test_pause_clears_event(self, commands_with_pause: Commands) -> None:
        result = commands_with_pause.dispatch("/pause")
        assert result is not None
        assert result.handled
        assert not commands_with_pause._pause_event.is_set()

    def test_resume_sets_event(self, commands_with_pause: Commands) -> None:
        commands_with_pause.dispatch("/pause")
        result = commands_with_pause.dispatch("/resume")
        assert result is not None
        assert result.handled
        assert commands_with_pause._pause_event.is_set()

    def test_resume_when_not_paused(self, commands_with_pause: Commands) -> None:
        result = commands_with_pause.dispatch("/resume")
        assert result is not None
        assert result.handled

    def test_guidance_injects_message(
        self, commands_with_pause: Commands, conversation: Conversation
    ) -> None:
        commands_with_pause.dispatch("/pause")
        result = commands_with_pause.dispatch("/guidance Try a different approach")
        assert result is not None
        assert result.handled
        # Check guidance was injected
        messages = conversation.messages
        guidance_found = any(
            "different approach" in msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        )
        assert guidance_found
        # Should also resume
        assert commands_with_pause._pause_event.is_set()

    def test_guidance_no_args(self, commands_with_pause: Commands) -> None:
        result = commands_with_pause.dispatch("/guidance")
        assert result is not None
        assert result.handled


def test_pause_without_event(commands: Commands) -> None:
    """Commands without pause_event should handle gracefully."""
    result = commands.dispatch("/pause")
    assert result is not None
    assert result.handled


class TestModelCommand:
    """Test /model command for switching models."""

    def test_model_shows_current(self, commands: Commands) -> None:
        """Test /model without args shows current model."""
        result = commands.dispatch("/model")
        assert result is not None
        assert result.handled

    def test_model_switches_valid(self, commands: Commands) -> None:
        """Test /model with valid provider-prefixed model."""
        result = commands.dispatch("/model ollama/qwen3")
        assert result is not None
        assert result.handled
        assert commands._llm_client.model == "ollama/qwen3"

    def test_model_rejects_no_slash(self, commands: Commands) -> None:
        """Test /model rejects model names without provider prefix."""
        result = commands.dispatch("/model qwen 3.5")
        assert result is not None
        assert result.handled
        # Model should NOT have changed
        assert commands._llm_client.model == "test-model"

    def test_model_rejects_bare_name(self, commands: Commands) -> None:
        """Test /model rejects bare model names."""
        result = commands.dispatch("/model claude")
        assert result is not None
        assert result.handled
        # Model should NOT have changed
        assert commands._llm_client.model == "test-model"

    def test_model_accepts_complex_path(self, commands: Commands) -> None:
        """Test /model accepts complex provider paths."""
        result = commands.dispatch("/model nvidia_nim/qwen/qwen3.5-397b-a17b")
        assert result is not None
        assert result.handled
        assert commands._llm_client.model == "nvidia_nim/qwen/qwen3.5-397b-a17b"
