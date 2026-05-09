"""Tests for skill command registration and dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.skills.commands import register_skill_commands
from godspeed.skills.loader import Skill, SkillSecurityError


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    trigger: str = "test",
    content: str = "Do the test thing.",
) -> Skill:
    return Skill(
        name=name,
        description=description,
        trigger=trigger,
        content=content,
        path=Path(f"/fake/{name}/SKILL.md"),
    )


class TestRegisterSkillCommands:
    """Test register_skill_commands()."""

    def test_registers_trigger_command(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        skills = [_make_skill()]
        register_skill_commands(commands, conversation, skills)

        calls = commands.register.call_args_list
        registered_names = [c[0][0] for c in calls]
        assert "/test" in registered_names
        assert "/skills" in registered_names

    def test_trigger_injects_message(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        skill = _make_skill(name="review", trigger="review", content="Review the code.")
        register_skill_commands(commands, conversation, [skill])

        handler = None
        for call in commands.register.call_args_list:
            if call[0][0] == "/review":
                handler = call[0][1]
                break

        assert handler is not None
        result = handler()

        conversation.add_user_message.assert_called_once()
        msg = conversation.add_user_message.call_args[0][0]
        assert "[Skill: review]" in msg
        assert "Review the code." in msg
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

        registered_names = [c[0][0] for c in commands.register.call_args_list]
        assert "/skills" in registered_names
        assert "/skill" in registered_names
        assert "/skill-evolve" in registered_names
        assert "/skill-dream" in registered_names
        assert len(registered_names) == 4

    def test_skills_command_handler_returns_handled(self) -> None:
        commands = MagicMock()
        conversation = MagicMock()
        register_skill_commands(commands, conversation, [_make_skill()])

        handler = None
        for call in commands.register.call_args_list:
            if call[0][0] == "/skills":
                handler = call[0][1]
                break

        assert handler is not None
        result = handler()
        assert result.handled is True


# ── /skill sub-command tests ──────────────────────────────────────────


class TestSkillSubCommands:
    """Test the /skill sub-command dispatcher."""

    @pytest.fixture
    def commands(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def conversation(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def handler(self, commands: MagicMock, conversation: MagicMock) -> callable:
        """Return the /skill handler function."""
        register_skill_commands(commands, conversation, [_make_skill()])
        for call in commands.register.call_args_list:
            if call[0][0] == "/skill":
                return call[0][1]
        msg = "/skill handler not registered"
        raise AssertionError(msg)

    def test_skill_list(self, handler: callable) -> None:
        result = handler("list")
        assert result is not None

    def test_skill_list_with_extra_args(self, handler: callable) -> None:
        result = handler("list --all")
        assert result is not None

    def test_skill_no_subcommand(self, handler: callable) -> None:
        result = handler("")
        assert result is not None

    def test_skill_unknown_subcommand(self, handler: callable) -> None:
        result = handler("unknown-sub")
        assert result is not None

    def test_skill_install_missing_dir(self, handler: callable) -> None:
        result = handler("install /nonexistent/path")
        assert result is not None

    def test_skill_install_success(self, handler: callable, tmp_path: Path) -> None:
        # Re-register with a mock hub that succeeds
        commands = MagicMock()
        conversation = MagicMock()
        mock_hub = MagicMock()
        mock_hub.install.return_value = _make_skill()
        register_skill_commands(
            commands,
            conversation,
            [_make_skill()],
            hub=mock_hub,
        )
        for call in commands.register.call_args_list:
            if call[0][0] == "/skill":
                call[0][1]
                break
        else:
            msg = "/skill handler not registered"
            raise AssertionError(msg)
        skill_dir = tmp_path / "myskill"
        skill_dir.mkdir(parents=True)
        result = handler(f"install {skill_dir}")
        assert result is not None

    def test_skill_install_security_fail(self, handler: callable, tmp_path: Path) -> None:
        # Re-register with a mock hub that raises SkillSecurityError
        commands = MagicMock()
        conversation = MagicMock()
        mock_hub = MagicMock()
        mock_hub.install.side_effect = SkillSecurityError("security fail")
        register_skill_commands(
            commands,
            conversation,
            [_make_skill()],
            hub=mock_hub,
        )
        for call in commands.register.call_args_list:
            if call[0][0] == "/skill":
                h = call[0][1]
                break
        else:
            msg = "/skill handler not registered"
            raise AssertionError(msg)
        skill_dir = tmp_path / "badskill"
        skill_dir.mkdir(parents=True)
        result = h(f"install {skill_dir}")
        assert result is not None

    def test_skill_remove_missing_name(self, handler: callable) -> None:
        result = handler("remove")
        assert result is not None

    @patch("godspeed.skills.commands.SkillHub.remove")
    def test_skill_remove_success(self, mock_remove: MagicMock, handler: callable) -> None:
        result = handler("remove myskill")
        assert result is not None
        mock_remove.assert_called_once_with("myskill")

    def test_skill_scan_no_arg(self, handler: callable) -> None:
        result = handler("scan")
        assert result is not None

    @patch("godspeed.skills.commands.scan_skill", return_value=[])
    def test_skill_scan_clean(
        self, mock_scan: MagicMock, handler: callable, tmp_path: Path
    ) -> None:
        result = handler(f"scan {tmp_path}")
        assert result is not None

    @patch("godspeed.skills.commands.scan_skill", return_value=["dangerous-shell"])
    def test_skill_scan_dangerous(self, mock_scan: MagicMock, handler: callable) -> None:
        result = handler("scan /some/path")
        assert result is not None

    def test_skill_verify_missing_name(self, handler: callable) -> None:
        result = handler("verify")
        assert result is not None

    @patch("godspeed.skills.commands.SkillHub.verify_integrity", return_value=True)
    def test_skill_verify_ok(self, mock_verify: MagicMock, handler: callable) -> None:
        result = handler("verify myskill")
        assert result is not None
        mock_verify.assert_called_once_with("myskill")

    @patch("godspeed.skills.commands.SkillHub.verify_integrity", return_value=False)
    def test_skill_verify_fail(self, mock_verify: MagicMock, handler: callable) -> None:
        result = handler("verify myskill")
        assert result is not None
        mock_verify.assert_called_once_with("myskill")

    def test_skill_hub_empty(self, handler: callable) -> None:
        result = handler("hub")
        assert result is not None

    @patch(
        "godspeed.skills.commands.SkillHub.list_installed",
        return_value=[{"name": "s1", "version": "1.0", "installed_at": "now"}],
    )
    def test_skill_hub_with_items(self, mock_list: MagicMock, handler: callable) -> None:
        result = handler("hub")
        assert result is not None

    def test_skill_generate_missing_topic(self, handler: callable) -> None:
        result = handler("generate")
        assert result is not None

    @patch("godspeed.skills.commands.WikiBridge")
    def test_skill_generate_success(
        self, mock_bridge_cls: MagicMock, handler: callable, tmp_path: Path
    ) -> None:
        mock_bridge = MagicMock()
        mock_bridge.generate_skill.return_value = tmp_path / "generated-skill"
        mock_bridge_cls.return_value = mock_bridge
        result = handler("generate nvfp4-benchmarks")
        assert result is not None
        mock_bridge.generate_skill.assert_called_once_with("nvfp4-benchmarks", output_name=None)

    @patch("godspeed.skills.commands.WikiBridge")
    def test_skill_generate_with_custom_name(
        self, mock_bridge_cls: MagicMock, handler: callable, tmp_path: Path
    ) -> None:
        mock_bridge = MagicMock()
        mock_bridge.generate_skill.return_value = tmp_path / "generated-skill"
        mock_bridge_cls.return_value = mock_bridge
        result = handler("generate nvfp4 --as nvfp4-bench")
        assert result is not None
        mock_bridge.generate_skill.assert_called_once_with("nvfp4", output_name="nvfp4-bench")

    @patch("godspeed.skills.commands.WikiBridge")
    def test_skill_generate_no_result(self, mock_bridge_cls: MagicMock, handler: callable) -> None:
        mock_bridge = MagicMock()
        mock_bridge.generate_skill.return_value = None
        mock_bridge_cls.return_value = mock_bridge
        result = handler("generate nonexistent")
        assert result is not None
        mock_bridge.generate_skill.assert_called_once_with("nonexistent", output_name=None)


# ── /skill-evolve tests ──────────────────────────────────────────────


class TestSkillEvolve:
    @pytest.fixture
    def commands(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def conversation(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def handler(self, commands: MagicMock, conversation: MagicMock, tmp_path: Path) -> callable:
        """Return /skill-evolve handler with a temp skills dir."""
        register_skill_commands(
            commands,
            conversation,
            [],
            skills_dir=tmp_path / ".godspeed" / "skills",
        )
        for call in commands.register.call_args_list:
            if call[0][0] == "/skill-evolve":
                return call[0][1]
        msg = "/skill-evolve handler not registered"
        raise AssertionError(msg)

    def test_evolve_missing_name(self, handler: callable) -> None:
        result = handler("")
        assert result is not None

    def test_evolve_nonexistent_skill(self, handler: callable) -> None:
        result = handler("nonexistent")
        assert result is not None

    @patch("godspeed.skills.commands.SkillEvolution.evolve", return_value=True)
    def test_evolve_success(
        self, mock_evolve: MagicMock, handler: callable, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / ".godspeed" / "skills" / "myskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: myskill\ndescription: test\n---\nbody")
        result = handler("myskill")
        assert result is not None

    @patch("godspeed.skills.commands.SkillEvolution.evolve", return_value=False)
    def test_evolve_no_eligible_lessons(
        self, mock_evolve: MagicMock, handler: callable, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / ".godspeed" / "skills" / "myskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: myskill\ndescription: test\n---\nbody")
        result = handler("myskill")
        assert result is not None


# ── /skill-dream tests ────────────────────────────────────────────────


class TestSkillDream:
    @pytest.fixture
    def commands(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def conversation(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def handler(self, commands: MagicMock, conversation: MagicMock, tmp_path: Path) -> callable:
        """Return /skill-dream handler."""
        register_skill_commands(
            commands,
            conversation,
            [],
            skills_dir=tmp_path / ".godspeed" / "skills",
        )
        for call in commands.register.call_args_list:
            if call[0][0] == "/skill-dream":
                return call[0][1]
        msg = "/skill-dream handler not registered"
        raise AssertionError(msg)

    @patch(
        "godspeed.skills.commands.SkillDream.run",
        return_value={"dates_normalized": 3, "errors": 0},
    )
    def test_dream_success(self, mock_run: MagicMock, handler: callable) -> None:
        result = handler("")
        assert result is not None
        mock_run.assert_called_once()
