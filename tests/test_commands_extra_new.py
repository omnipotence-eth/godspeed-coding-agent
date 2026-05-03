"""Additional tests for tui/commands.py to increase coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tui.commands import CommandResult, Commands


@pytest.fixture
def mock_conversation():
    return MagicMock()


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.model = "test-model"
    client.fallback_models = ["fallback-model"]
    return client


@pytest.fixture
def mock_permission_engine():
    engine = MagicMock()
    engine.deny_rules = []
    engine.allow_rules = []
    engine.ask_rules = []
    engine.session_grants = []
    return engine


@pytest.fixture
def mock_audit_trail(tmp_path: Path):
    trail = MagicMock()
    trail.record_count = 42
    trail.log_path = tmp_path / "test.audit.jsonl"
    trail.verify_chain.return_value = (True, "Chain valid")
    return trail


@pytest.fixture
def commands(
    mock_conversation,
    mock_llm_client,
    mock_permission_engine,
    mock_audit_trail,
    tmp_path: Path,
):
    return Commands(
        conversation=mock_conversation,
        llm_client=mock_llm_client,
        permission_engine=mock_permission_engine,
        audit_trail=mock_audit_trail,
        session_id="test-session-1234",
        cwd=tmp_path,
    )


class TestRegister:
    def test_register_with_slash(self, commands):
        """Test registering a command with leading slash."""

        def handler(args):
            return CommandResult(message="test")

        commands.register("test_cmd", handler)
        assert "/test_cmd" in commands._handlers

    def test_register_without_slash(self, commands):
        """Test registering a command without leading slash."""

        def handler(args):
            return CommandResult(message="test")

        commands.register("test_cmd", handler)
        assert "/test_cmd" in commands._handlers


class TestDispatch:
    def test_dispatch_non_command(self, commands):
        """Test dispatching non-command input."""
        result = commands.dispatch("hello world")
        assert result is None

    def test_dispatch_unknown_command(self, commands):
        """Test dispatching unknown command."""
        result = commands.dispatch("/nonexistent")
        assert result is not None
        assert result.handled is True

    def test_dispatch_known_command(self, commands):
        """Test dispatching known command."""

        def test_handler(args):
            return CommandResult(message="handled")

        commands._handlers["/test"] = test_handler
        result = commands.dispatch("/test arg1")
        assert result is not None
        assert result.message == "handled"

    def test_dispatch_with_args(self, commands):
        """Test dispatching command with arguments."""

        def test_handler(args):
            return CommandResult(message=f"args: {args}")

        commands._handlers["/test"] = test_handler
        result = commands.dispatch("/test some args")
        assert result is not None
        assert "some args" in result.message


class TestCmdClear:
    def test_clear_calls_conversation_clear(self, commands, mock_conversation):
        """Test /clear command calls conversation.clear()."""
        result = commands._cmd_clear()
        mock_conversation.clear.assert_called_once()
        assert result.handled is True


class TestCmdUndo:
    def test_undo_not_git_repo(self, commands):
        """Test /undo when not in a git repo."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1  # git log fails
            result = commands._cmd_undo()
            assert result.handled is True

    def test_undo_success(self, commands):
        """Test /undo successful case."""
        with patch("subprocess.run") as mock_run:
            # First call (git log) succeeds
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123 Commit message\n"),
                MagicMock(returncode=0, stderr=""),
            ]
            result = commands._cmd_undo()
            assert result.handled is True

    def test_undo_git_reset_fails(self, commands):
        """Test /undo when git reset fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123 Commit message\n"),
                MagicMock(returncode=1, stderr="error message"),
            ]
            result = commands._cmd_undo()
            assert result.handled is True

    def test_undo_timeout(self, commands):
        """Test /undo when git command times out."""
        with patch("subprocess.run") as mock_run:
            from subprocess import TimeoutExpired

            mock_run.side_effect = TimeoutExpired(cmd="git", timeout=10)
            result = commands._cmd_undo()
            assert result.handled is True

    def test_undo_file_not_found(self, commands):
        """Test /undo when git is not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = commands._cmd_undo()
            assert result.handled is True


class TestCmdAudit:
    def test_audit_trail_disabled(
        self,
        mock_conversation,
        mock_llm_client,
        mock_permission_engine,
        tmp_path,
    ):
        """Test /audit when audit trail is disabled."""
        commands = Commands(
            conversation=mock_conversation,
            llm_client=mock_llm_client,
            permission_engine=mock_permission_engine,
            audit_trail=None,
            session_id="test",
            cwd=tmp_path,
        )
        result = commands._cmd_audit()
        assert result.handled is True

    def test_audit_trail_enabled(self, commands):
        """Test /audit when audit trail is enabled."""
        result = commands._cmd_audit()
        assert result.handled is True
        commands._audit_trail.verify_chain.assert_called_once()


class TestCmdPermissions:
    def test_permissions_disabled(
        self,
        mock_conversation,
        mock_llm_client,
        mock_audit_trail,
        tmp_path,
    ):
        """Test /permissions when permission engine is disabled."""
        commands = Commands(
            conversation=mock_conversation,
            llm_client=mock_llm_client,
            permission_engine=None,
            audit_trail=mock_audit_trail,
            session_id="test",
            cwd=tmp_path,
        )
        result = commands._cmd_permissions()
        assert result.handled is True

    def test_permissions_enabled(self, commands):
        """Test /permissions when permission engine is enabled."""
        result = commands._cmd_permissions()
        assert result.handled is True


class TestCmdRemember:
    def test_remember_no_args(self, commands):
        """Test /remember with no arguments."""
        result = commands._cmd_remember("")
        assert result.handled is True

    def test_remember_insufficient_args(self, commands):
        """Test /remember with insufficient arguments."""
        result = commands._cmd_remember("approve")
        assert result.handled is True

    def test_remember_unknown_action(self, commands):
        """Test /remember with unknown action."""
        result = commands._cmd_remember("invalid_action Shell(*)")
        assert result.handled is True

    def test_remember_invalid_pattern(self, commands):
        """Test /remember with invalid pattern (no parentheses)."""
        result = commands._cmd_remember("approve shell*")
        assert result.handled is True

    def test_remember_valid_pattern(self, commands, tmp_path: Path):
        """Test /remember with valid pattern."""
        with patch("godspeed.tui.commands.append_permission_rule") as mock_append:
            mock_append.return_value = tmp_path / "settings.yaml"
            commands._permission_engine.add_rule = MagicMock()
            result = commands._cmd_remember("approve Shell(pytest *)")
            assert result.handled is True

    def test_remember_with_project_flag(self, commands, tmp_path: Path):
        """Test /remember with --project flag."""
        with patch("godspeed.tui.commands.append_permission_rule") as mock_append:
            mock_append.return_value = tmp_path / ".godspeed" / "settings.yaml"
            commands._permission_engine.add_rule = MagicMock()
            result = commands._cmd_remember("approve Shell(pytest *) --project")
            assert result.handled is True

    def test_remember_write_fails(self, commands, tmp_path: Path):
        """Test /remember when writing to settings fails."""
        with patch("godspeed.tui.commands.append_permission_rule") as mock_append:
            mock_append.return_value = tmp_path / "settings.yaml"
            result = commands._cmd_remember("approve Shell(pytest *)")
            assert result.handled is True

    def test_remember_add_rule_fails(self, commands, tmp_path: Path):
        """Test /remember when adding rule to engine fails."""
        with patch("godspeed.tui.commands.append_permission_rule") as mock_append:
            mock_append.return_value = tmp_path / "settings.yaml"
            commands._permission_engine.add_rule = MagicMock(side_effect=ValueError("Invalid"))
            result = commands._cmd_remember("approve Shell(pytest *)")
            assert result.handled is True
