"""Tests for the /remember slash command + its write-back helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from godspeed.agent.conversation import Conversation
from godspeed.config import append_permission_rule
from godspeed.security.permissions import PermissionEngine
from godspeed.tools.base import RiskLevel
from godspeed.tui.commands import Commands


@pytest.fixture
def engine() -> PermissionEngine:
    return PermissionEngine(
        deny_patterns=["FileRead(.env)"],
        allow_patterns=["Shell(git *)"],
        ask_patterns=["Shell(*)"],
        tool_risk_levels={"shell": RiskLevel.HIGH, "file_read": RiskLevel.READ_ONLY},
    )


@pytest.fixture
def commands(tmp_path: Path, engine: PermissionEngine) -> Commands:
    conv = Conversation("You are a coding agent.", max_tokens=100_000)
    llm = MagicMock()
    llm.model = "test-model"
    llm.fallback_models = []
    llm.total_input_tokens = 0
    llm.total_output_tokens = 0
    return Commands(
        conversation=conv,
        llm_client=llm,
        permission_engine=engine,
        audit_trail=None,
        session_id="test-session",
        cwd=tmp_path,
    )


class TestAppendPermissionRule:
    """The YAML write-back helper underlying /remember."""

    def test_writes_allow_rule_to_project_file(self, tmp_path: Path) -> None:
        result = append_permission_rule("Shell(pytest *)", "allow", project_dir=tmp_path)
        assert result == tmp_path / ".godspeed" / "settings.yaml"
        data = yaml.safe_load(result.read_text())
        assert data["permissions"]["allow"] == ["Shell(pytest *)"]

    def test_writes_deny_rule(self, tmp_path: Path) -> None:
        result = append_permission_rule("FileWrite(*.env*)", "deny", project_dir=tmp_path)
        assert result is not None
        data = yaml.safe_load(result.read_text())
        assert data["permissions"]["deny"] == ["FileWrite(*.env*)"]

    def test_writes_ask_rule(self, tmp_path: Path) -> None:
        result = append_permission_rule("Shell(rm *)", "ask", project_dir=tmp_path)
        assert result is not None
        data = yaml.safe_load(result.read_text())
        assert data["permissions"]["ask"] == ["Shell(rm *)"]

    def test_duplicate_pattern_is_silently_idempotent(self, tmp_path: Path) -> None:
        append_permission_rule("Shell(pytest *)", "allow", project_dir=tmp_path)
        append_permission_rule("Shell(pytest *)", "allow", project_dir=tmp_path)
        settings_path = tmp_path / ".godspeed" / "settings.yaml"
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["Shell(pytest *)"]  # no duplicate

    def test_preserves_other_yaml_content(self, tmp_path: Path) -> None:
        settings_path = tmp_path / ".godspeed" / "settings.yaml"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            yaml.safe_dump({"model": "claude-sonnet-4", "max_context_tokens": 200_000})
        )
        append_permission_rule("Shell(make)", "allow", project_dir=tmp_path)
        data = yaml.safe_load(settings_path.read_text())
        assert data["model"] == "claude-sonnet-4"
        assert data["max_context_tokens"] == 200_000
        assert data["permissions"]["allow"] == ["Shell(make)"]

    def test_invalid_action_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="action must be"):
            append_permission_rule("x", "forbid", project_dir=tmp_path)


class TestPermissionEngineAddRule:
    """In-memory rule injection via add_rule."""

    def test_add_allow_rule_takes_effect(self, engine: PermissionEngine) -> None:
        engine.add_rule("Shell(make)", "allow")
        assert any(r.pattern == "Shell(make)" for r in engine.allow_rules)

    def test_add_deny_rule_takes_effect(self, engine: PermissionEngine) -> None:
        engine.add_rule("Shell(rm -rf *)", "deny")
        assert any(r.pattern == "Shell(rm -rf *)" for r in engine.deny_rules)

    def test_add_ask_rule_takes_effect(self, engine: PermissionEngine) -> None:
        engine.add_rule("FileWrite(*)", "ask")
        assert any(r.pattern == "FileWrite(*)" for r in engine.ask_rules)

    def test_add_rule_invalid_action_raises(self, engine: PermissionEngine) -> None:
        with pytest.raises(ValueError, match="action must be"):
            engine.add_rule("Shell(x)", "forbid")


class TestRememberCommand:
    """End-to-end tests for the /remember slash command."""

    def test_approve_alias_persists_as_allow(self, commands: Commands, tmp_path: Path) -> None:
        # Patch append_permission_rule to write to tmp_path rather than
        # the real ~/.godspeed (global scope path would otherwise leak).
        with patch(
            "godspeed.tui.commands.append_permission_rule",
            wraps=append_permission_rule,
        ) as spy:
            result = commands.dispatch("/remember approve Shell(pytest *)")
            assert result is not None and result.handled

        # Called with action="allow" (approve is an alias).
        assert spy.called
        kwargs = spy.call_args.kwargs
        assert kwargs["action"] == "allow"
        assert kwargs["pattern"] == "Shell(pytest *)"
        assert kwargs["project_dir"] is None  # global by default

    def test_deny_action_persists_as_deny(self, commands: Commands) -> None:
        with patch(
            "godspeed.tui.commands.append_permission_rule",
            wraps=append_permission_rule,
        ) as spy:
            commands.dispatch("/remember deny FileWrite(*.env*)")
        assert spy.call_args.kwargs["action"] == "deny"
        assert spy.call_args.kwargs["pattern"] == "FileWrite(*.env*)"

    def test_project_flag_scopes_to_cwd(self, commands: Commands, tmp_path: Path) -> None:
        commands.dispatch("/remember approve Shell(make) --project")
        settings_path = tmp_path / ".godspeed" / "settings.yaml"
        assert settings_path.exists()
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["Shell(make)"]

    def test_rule_takes_effect_in_session(
        self, commands: Commands, engine: PermissionEngine
    ) -> None:
        # Baseline — Shell(pytest *) is not in the allow list yet.
        assert not any(r.pattern == "Shell(pytest *)" for r in engine.allow_rules)

        commands.dispatch("/remember approve Shell(pytest *) --project")

        # After /remember, the rule is live in the session's engine.
        assert any(r.pattern == "Shell(pytest *)" for r in engine.allow_rules)

    def test_no_args_shows_usage(self, commands: Commands) -> None:
        result = commands.dispatch("/remember")
        assert result is not None and result.handled
        # No persistence attempted.

    def test_unknown_action_is_rejected(self, commands: Commands) -> None:
        with patch("godspeed.tui.commands.append_permission_rule") as spy:
            result = commands.dispatch("/remember yeet Shell(rm *)")
        assert result is not None and result.handled
        spy.assert_not_called()  # nothing persisted for bogus action

    def test_pattern_without_parens_is_rejected(self, commands: Commands) -> None:
        with patch("godspeed.tui.commands.append_permission_rule") as spy:
            result = commands.dispatch("/remember approve notAPattern")
        assert result is not None and result.handled
        spy.assert_not_called()

    def test_pattern_with_spaces_is_preserved(self, commands: Commands, tmp_path: Path) -> None:
        # Pattern "Shell(git commit *)" contains an internal space —
        # token rejoin must preserve it.
        commands.dispatch("/remember approve Shell(git commit *) --project")
        settings_path = tmp_path / ".godspeed" / "settings.yaml"
        data = yaml.safe_load(settings_path.read_text())
        assert data["permissions"]["allow"] == ["Shell(git commit *)"]

    def test_no_permission_engine_shows_error(self, tmp_path: Path) -> None:
        # Simulate a bare session without a permission engine wired up.
        conv = Conversation("sys", max_tokens=100_000)
        llm = MagicMock()
        llm.model = "m"
        llm.fallback_models = []
        llm.total_input_tokens = 0
        llm.total_output_tokens = 0
        bare = Commands(
            conversation=conv,
            llm_client=llm,
            permission_engine=None,
            audit_trail=None,
            session_id="s",
            cwd=tmp_path,
        )
        with patch("godspeed.tui.commands.append_permission_rule") as spy:
            result = bare.dispatch("/remember approve Shell(x)")
        assert result is not None and result.handled
        spy.assert_not_called()
