"""Tests for skill command registration and dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from godspeed.skills.commands import register_skill_commands
from godspeed.skills.loader import SkillDefinition


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    trigger: str = "test",
    content: str = "Do the test thing.",
) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=description,
        trigger=trigger,
        content=content,
    )


class TestRegisterSkillCommands:
    """Test register_skill_commands()."""

    def test_registers_trigger_command(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        skills = [_make_skill()]
        register_skill_commands(commands, conversation, skills)

        # Should register /test and /skills
        calls = commands.register.call_args_list
        registered_names = [c[0][0] for c in calls]
        assert "/test" in registered_names
        assert "/skills" in registered_names

    def test_trigger_injects_message(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        skill = _make_skill(name="review", trigger="review", content="Review the code.")
        register_skill_commands(commands, conversation, [skill])

        # Get the handler that was registered for /review
        handler = None
        for call in commands.register.call_args_list:
            if call[0][0] == "/review":
                handler = call[0][1]
                break

        assert handler is not None
        result = handler()

        # Should inject skill content into conversation
        conversation.add_user_message.assert_called_once()
        msg = conversation.add_user_message.call_args[0][0]
        assert "[Skill: review]" in msg
        assert "Review the code." in msg

        # handled=False so agent_loop processes it
        assert result.handled is False

    def test_multiple_skills_registered(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        skills = [
            _make_skill(trigger="review", name="review"),
            _make_skill(trigger="test", name="test"),
            _make_skill(trigger="deploy", name="deploy"),
        ]
        register_skill_commands(commands, conversation, skills)

        registered_names = [c[0][0] for c in commands.register.call_args_list]
        assert "/review" in registered_names
        assert "/test" in registered_names
        assert "/deploy" in registered_names
        assert "/skills" in registered_names

    def test_skills_command_with_no_skills(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        register_skill_commands(commands, conversation, [])

        # Only /skills should be registered
        registered_names = [c[0][0] for c in commands.register.call_args_list]
        assert "/skills" in registered_names
        assert len(registered_names) == 1

    def test_skills_command_handler_returns_handled(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        register_skill_commands(commands, conversation, [_make_skill()])

        # Get /skills handler
        handler = None
        for call in commands.register.call_args_list:
            if call[0][0] == "/skills":
                handler = call[0][1]
                break

        assert handler is not None
        result = handler()
        assert result.handled is True
