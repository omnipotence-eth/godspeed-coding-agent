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

        commands.register("/test_cmd", handler)
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
        self, mock_conversation, mock_llm_client, mock_permission_engine, tmp_path,
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


class TestCmdPermissions:
    def test_permissions_disabled(
        self, mock_conversation, mock_llm_client, mock_audit_trail, tmp_path,
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


class TestCmdModel:
    def test_model_no_args(self, commands, mock_llm_client, tmp_path):
        """Test /model with no arguments."""
        mock_llm_client.model = "test-model"
        mock_llm_client.fallback_models = []
        result = commands._cmd_model("")
        assert result.handled is True

    def test_model_with_preset(self, commands, mock_llm_client):
        """Test /model with preset name."""
        mock_llm_client.model = "old-model"
        with patch("godspeed.config.GodspeedSettings.MODEL_PRESETS", {"coding": "ollama/qwen3:4b"}):
            result = commands._cmd_model("coding")
            assert result.handled is True
            assert mock_llm_client.model == "ollama/qwen3:4b"

    def test_model_with_unknown_preset(self, commands, mock_llm_client):
        """Test /model with unknown preset."""
        mock_llm_client.model = "some-model"
        result = commands._cmd_model("unknown_preset")
        assert result.handled is True
        assert mock_llm_client.model == "unknown_preset"

    def test_model_with_direct_name(self, commands, mock_llm_client):
        """Test /model with direct model name."""
        mock_llm_client.model = "old-model"
        result = commands._cmd_model("new-model")
        assert result.handled is True
        assert mock_llm_client.model == "new-model"

    def test_model_ollama_preset_installed(self, commands, mock_llm_client):
        """Test /model with ollama preset, model installed."""
        mock_llm_client.model = "old-model"
        with patch("godspeed.config.GodspeedSettings.MODEL_PRESETS", {"coding": "ollama/qwen3:4b"}):
            with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=True):
                result = commands._cmd_model("coding")
                assert result.handled is True

    def test_model_ollama_preset_not_installed(self, commands, mock_llm_client):
        """Test /model with ollama preset, model not installed."""
        mock_llm_client.model = "old-model"
        with patch("godspeed.config.GodspeedSettings.MODEL_PRESETS", {"coding": "ollama/qwen3:4b"}):
            with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=False):
                result = commands._cmd_model("coding")
                assert result.handled is True

    def test_model_direct_ollama_installed(self, commands, mock_llm_client):
        """Test /model with direct ollama model name, installed."""
        mock_llm_client.model = "old-model"
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=True):
            result = commands._cmd_model("ollama/qwen3:4b")
            assert result.handled is True
            assert mock_llm_client.model == "ollama/qwen3:4b"

    def test_model_direct_ollama_not_installed(self, commands, mock_llm_client):
        """Test /model with direct ollama model name, not installed."""
        mock_llm_client.model = "old-model"
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=False):
            result = commands._cmd_model("ollama/qwen3:4b")
            assert result.handled is True

    def test_model_with_fallbacks(self, commands, mock_llm_client):
        """Test /model when fallbacks are configured."""
        mock_llm_client.model = "test-model"
        mock_llm_client.fallback_models = ["fallback1", "fallback2"]
        result = commands._cmd_model("")
        assert result.handled is True

    def test_model_no_preset_match(self, commands, mock_llm_client):
        """Test /model with no preset match (else branch)."""
        mock_llm_client.model = "old-model"
        result = commands._cmd_model("new-model")
        assert result.handled is True
        assert mock_llm_client.model == "new-model"

    def test_model_ollama_direct_no_preset(self, commands, mock_llm_client):
        """Test /model with ollama/ direct name, no preset."""
        mock_llm_client.model = "old-model"
        with patch("godspeed.tools.ollama_manager.is_model_installed", return_value=False):
            result = commands._cmd_model("ollama/codellama:latest")
            assert result.handled is True


class TestCmdHelp:
    def test_help_command(self, commands):
        """Test /help command."""
        result = commands._cmd_help()
        assert result.handled is True


class TestCmdQuit:
    def test_quit_command(self, commands):
        """Test /quit command."""
        result = commands._cmd_quit()
        assert result.handled is True
        assert result.should_quit is True


class TestCmdExtend:
    def test_extend_no_args(self, commands):
        """Test /extend with no arguments."""
        commands.max_iterations = None
        result = commands._cmd_extend("")
        assert result.handled is True

    def test_extend_with_number(self, commands):
        """Test /extend with number argument."""
        result = commands._cmd_extend("100")
        assert result.handled is True
        assert commands.max_iterations == 100

    def test_extend_with_invalid_number(self, commands):
        """Test /extend with invalid number."""
        result = commands._cmd_extend("not_a_number")
        assert result.handled is True


class TestCmdAutocommit:
    def test_autocommit_no_args(self, commands):
        """Test /autocommit with no arguments."""
        result = commands._cmd_autocommit("")
        assert result.handled is True

    def test_autocommit_on(self, commands):
        """Test /autocommit on."""
        result = commands._cmd_autocommit("on")
        assert result.handled is True
        assert commands.auto_commit is True

    def test_autocommit_off(self, commands):
        """Test /autocommit off."""
        result = commands._cmd_autocommit("off")
        assert result.handled is True
        assert commands.auto_commit is False

    def test_autocommit_with_number(self, commands):
        """Test /autocommit with number."""
        result = commands._cmd_autocommit("10")
        assert result.handled is True
        assert commands.auto_commit_threshold == 10


class TestCmdArchitect:
    def test_architect_toggle(self, commands):
        """Test /architect command toggles mode."""
        initial = commands.architect_mode
        result = commands._cmd_architect()
        assert result.handled is True
        assert commands.architect_mode != initial


class TestCmdThink:
    def test_think_no_args_toggle_on(self, commands):
        """Test /think with no args when thinking is off."""
        commands._llm_client.thinking_budget = 0
        result = commands._cmd_think("")
        assert result.handled is True

    def test_think_no_args_toggle_off(self, commands):
        """Test /think with no args when thinking is on."""
        commands._llm_client.thinking_budget = 10000
        result = commands._cmd_think("")
        assert result.handled is True

    def test_think_off(self, commands):
        """Test /think off."""
        commands._llm_client.thinking_budget = 10000
        result = commands._cmd_think("off")
        assert result.handled is True
        assert commands._llm_client.thinking_budget == 0

    def test_think_with_budget(self, commands):
        """Test /think with budget."""
        result = commands._cmd_think("5000")
        assert result.handled is True

    def test_think_invalid_budget(self, commands):
        """Test /think with invalid budget."""
        result = commands._cmd_think("invalid")
        assert result.handled is True

    def test_think_budget_too_low(self, commands):
        """Test /think with budget below 1000."""
        result = commands._cmd_think("500")
        assert result.handled is True


class TestCmdBudget:
    def test_budget_no_args(self, commands, tmp_path):
        """Test /budget with no args."""
        commands._llm_client.total_cost_usd = 5.0
        commands._llm_client.max_cost_usd = 10.0
        commands._llm_client.total_input_tokens = 1000
        commands._llm_client.total_output_tokens = 500
        commands._llm_client.model = "test-model"
        result = commands._cmd_budget("")
        assert result.handled is True

    def test_budget_no_limit(self, commands, tmp_path):
        """Test /budget when no limit set."""
        commands._llm_client.total_cost_usd = 5.0
        commands._llm_client.max_cost_usd = 0.0
        commands._llm_client.total_input_tokens = 1000
        commands._llm_client.total_output_tokens = 500
        commands._llm_client.model = "test-model"
        result = commands._cmd_budget("")
        assert result.handled is True

    def test_budget_off(self, commands):
        """Test /budget off."""
        result = commands._cmd_budget("off")
        assert result.handled is True
        assert commands._llm_client.max_cost_usd == 0.0

    def test_budget_unlimited(self, commands):
        """Test /budget unlimited."""
        result = commands._cmd_budget("unlimited")
        assert result.handled is True

    def test_budget_zero(self, commands):
        """Test /budget 0."""
        result = commands._cmd_budget("0")
        assert result.handled is True

    def test_budget_with_dollar_sign(self, commands):
        """Test /budget with $ prefix."""
        result = commands._cmd_budget("$50.00")
        assert result.handled is True

    def test_budget_invalid(self, commands):
        """Test /budget with invalid amount."""
        result = commands._cmd_budget("invalid")
        assert result.handled is True

    def test_budget_negative(self, commands):
        """Test /budget with negative amount."""
        result = commands._cmd_budget("-10")
        assert result.handled is True


class TestCmdEvolve:
    def test_evolve_no_args(self, commands):
        """Test /evolve with no args."""
        result = commands._cmd_evolve("")
        assert result.handled is True
