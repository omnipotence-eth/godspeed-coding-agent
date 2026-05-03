"""Tests for godspeed.skills.commands."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from godspeed.skills.commands import register_skill_commands
from godspeed.skills.loader import SkillDefinition


class TestRegisterSkillCommands:
    def test_register_single_skill(self):
        """Test that a skill gets registered as a slash command."""
        mock_commands = MagicMock()
        mock_conversation = MagicMock()
        skill = SkillDefinition(
            name="Test Skill",
            trigger="test-skill",
            description="A test skill",
            content="Test content",
        )

        register_skill_commands(mock_commands, mock_conversation, [skill])

        # Should register /test-skill command + /skills command
        assert mock_commands.register.call_count == 2
        # Check that /test-skill was registered
        registered_commands = [call[0][0] for call in mock_commands.register.call_args_list]
        assert "/test-skill" in registered_commands

    def test_register_multiple_skills(self):
        """Test that multiple skills get registered."""
        mock_commands = MagicMock()
        mock_conversation = MagicMock()
        skills = [
            SkillDefinition(
                name="Skill 1", trigger="skill1", description="First", content="Content 1"
            ),
            SkillDefinition(
                name="Skill 2", trigger="skill2", description="Second", content="Content 2"
            ),
        ]

        register_skill_commands(mock_commands, mock_conversation, skills)

        assert mock_commands.register.call_count == 3  # 2 skills + /skills

    def test_skill_handler_injects_message(self):
        """Test that activating a skill injects the content as a user message."""
        mock_commands = MagicMock()
        mock_conversation = MagicMock()
        skill = SkillDefinition(
            name="Test Skill",
            trigger="test",
            description="Test",
            content="Skill content here",
        )

        register_skill_commands(mock_commands, mock_conversation, [skill])

        # Get the registered handler
        call_args = mock_commands.register.call_args_list[0]
        handler = call_args[0][1]

        # Call the handler

        result = handler()

        # Should inject message and return handled=False
        mock_conversation.add_user_message.assert_called_once()
        injected = mock_conversation.add_user_message.call_args[0][0]
        assert "Skill: Test Skill" in injected
        assert "Skill content here" in injected
        assert result.handled is False

    def test_skills_command_no_skills(self):
        """Test /skills command when no skills are installed."""
        mock_commands = MagicMock()
        mock_conversation = MagicMock()

        register_skill_commands(mock_commands, mock_conversation, [])

        # Should register /skills command
        # Find the /skills registration
        for call in mock_commands.register.call_args_list:
            if call[0][0] == "/skills":
                handler = call[0][1]
                result = handler()
                assert result.handled is True
                break
        else:
            pytest.fail("Did not register /skills command")

    def test_skills_command_with_skills(self):
        """Test /skills command lists available skills."""
        mock_commands = MagicMock()
        mock_conversation = MagicMock()
        skills = [
            SkillDefinition(name="Skill 1", trigger="s1", description="First", content="C1"),
        ]

        register_skill_commands(mock_commands, mock_conversation, skills)

        # Find the /skills registration
        for call in mock_commands.register.call_args_list:
            if call[0][0] == "/skills":
                handler = call[0][1]
                result = handler()
                assert result.handled is True
                break
        else:
            pytest.fail("Did not register /skills command")
