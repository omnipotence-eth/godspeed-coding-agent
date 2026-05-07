"""Additional CLI tests covering remaining untested paths in cli.py."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from godspeed.cli import (
    _build_tool_registry,
    _ensure_llamacpp,
    _ensure_ollama,
    _is_ollama_running,
    _parse_env_file,
    _setup_logging,
)


class TestParseEnvFileEdgeCases:
    def test_non_existent_file_returns_empty(self, tmp_path: Path) -> None:
        assert _parse_env_file(tmp_path / "nonexistent.env") == {}

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        env = tmp_path / "empty.env"
        env.write_text("")
        assert _parse_env_file(env) == {}


class TestSetupLogging:
    def test_sets_godspeed_debug_when_verbose(self) -> None:
        _setup_logging(verbose=True)
        godspeed_logger = logging.getLogger("godspeed")
        assert godspeed_logger.level == logging.DEBUG

    def test_sets_godspeed_warning_when_not_verbose(self) -> None:
        _setup_logging(verbose=False)
        godspeed_logger = logging.getLogger("godspeed")
        assert godspeed_logger.level == logging.WARNING

    def test_root_logger_stays_warning(self) -> None:
        _setup_logging(verbose=True)
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestIsOllamaRunning:
    @patch("urllib.request.urlopen")
    def test_returns_true_when_reachable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__.return_value.status = 200
        assert _is_ollama_running() is True

    @patch("urllib.request.urlopen")
    def test_returns_false_on_exception(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = OSError("connection refused")
        assert _is_ollama_running() is False


class TestEnsureOllama:
    @patch("godspeed.cli._is_ollama_running")
    def test_returns_true_if_already_running(self, mock_is_running: MagicMock) -> None:
        mock_is_running.return_value = True
        assert _ensure_ollama() is True

    @patch("godspeed.cli._is_ollama_running")
    @patch("shutil.which")
    def test_returns_false_if_not_installed(
        self, mock_which: MagicMock, mock_is_running: MagicMock
    ) -> None:
        mock_is_running.return_value = False
        mock_which.return_value = None
        assert _ensure_ollama() is False

    @patch("godspeed.cli._is_ollama_running")
    @patch("shutil.which")
    @patch("subprocess.Popen")
    def test_starts_ollama_and_polls(
        self,
        mock_popen: MagicMock,
        mock_which: MagicMock,
        mock_is_running: MagicMock,
    ) -> None:
        mock_is_running.side_effect = [False, True]
        mock_which.return_value = "/usr/bin/ollama"
        assert _ensure_ollama() is True
        mock_popen.assert_called_once()


class TestEnsureLlamaCpp:
    @patch("godspeed.tools.llamacpp_manager.is_server_running")
    def test_returns_true_if_already_running(self, mock_is_running: MagicMock) -> None:
        mock_is_running.return_value = True
        assert _ensure_llamacpp() is True

    @patch("godspeed.tools.llamacpp_manager.is_server_running")
    @patch("godspeed.tools.llamacpp_manager.start_server")
    def test_starts_server_if_not_running(
        self, mock_start: MagicMock, mock_is_running: MagicMock
    ) -> None:
        mock_is_running.side_effect = [False, True]
        mock_start.return_value = MagicMock()
        assert _ensure_llamacpp() is True
        mock_start.assert_called_once()


class TestBuildToolRegistry:
    def test_builds_full_registry(self) -> None:
        registry, risk_levels = _build_tool_registry(tool_set="full")
        assert registry is not None
        assert len(risk_levels) > 10

    def test_builds_local_registry_without_web_tools(self) -> None:
        registry, _ = _build_tool_registry(tool_set="local")
        tool_names = [t.name for t in registry.list_tools()]
        assert "web_search" not in tool_names
        assert "web_fetch" not in tool_names

    def test_all_tools_have_risk_levels(self) -> None:
        registry, risk_levels = _build_tool_registry(tool_set="full")
        for tool in registry.list_tools():
            assert tool.name in risk_levels, f"{tool.name} missing risk level"


class TestCLICommands:
    def test_init_command(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0
        assert "Godspeed" in result.output or "Welcome" in result.output

    def test_run_command_no_task_shows_error(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        # Should error on missing task
        assert result.exit_code != 0

    @patch("godspeed.cli._headless_run")
    def test_run_command_with_task(self, mock_headless: MagicMock) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        mock_headless.return_value = 0
        with patch("asyncio.run", return_value=0):
            runner = CliRunner()
            result = runner.invoke(main, ["run", "test task"])
            assert result.exit_code == 0


    def test_verify_audit_no_dir(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["audit", "verify", "--audit-dir", r"C:\nonexistent"])
        assert result.exit_code != 0

    def test_verify_audit_with_logs(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        log_file = audit_dir / "test-session.audit.jsonl"
        log_file.write_text(
            '{"event_type":"session_start","session_id":"test-session"}\n'
        )
        with patch("godspeed.audit.trail.AuditTrail") as mock_trail_cls:
            mock_instance = MagicMock()
            mock_instance.verify_chain.return_value = (True, "valid chain")
            mock_trail_cls.return_value = mock_instance

            runner = CliRunner()
            result = runner.invoke(main, ["audit", "verify", "--audit-dir", str(audit_dir)])
            assert result.exit_code == 0

    def test_verify_audit_with_logs_broken(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        log_file = audit_dir / "test-session.audit.jsonl"
        log_file.write_text(
            '{"event_type":"session_start","session_id":"test-session"}\n'
        )
        with patch("godspeed.audit.trail.AuditTrail") as mock_trail_cls:
            mock_instance = MagicMock()
            mock_instance.verify_chain.return_value = (False, "hash mismatch")
            mock_trail_cls.return_value = mock_instance

            runner = CliRunner()
            result = runner.invoke(main, ["audit", "verify", "--audit-dir", str(audit_dir)])
            # Batch mode reports each log's status without sys.exit
            assert result.exit_code == 0
            assert "BROKEN" in result.output

    def test_verify_audit_no_logs_found(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, ["audit", "verify", "--audit-dir", str(audit_dir)])
        assert result.exit_code == 0

    def test_serve_command_calls_run_server(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        with patch("godspeed.mcp_server.server.run_server", return_value=0):
            runner = CliRunner()
            result = runner.invoke(main, ["serve"])
            assert result.exit_code == 0

    def test_serve_keyboard_interrupt(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        with patch("godspeed.mcp_server.server.run_server", side_effect=KeyboardInterrupt):
            runner = CliRunner()
            result = runner.invoke(main, ["serve"])
            assert result.exit_code == 0

    def test_serve_help(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output



    def test_headless_run_keyboard_interrupt(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        with patch("asyncio.run", side_effect=KeyboardInterrupt):
            runner = CliRunner()
            result = runner.invoke(main, ["run", "test task"])
            assert result.exit_code == 130

    def test_models_command(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["models", "--help"])
        assert result.exit_code == 0

    def test_main_help(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Godspeed" in result.output or "godspeed" in result.output

    def test_main_group(self) -> None:
        """Test main group invocation without command shows help."""
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0

    def test_audit_verify_single_session_valid(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        log_file = audit_dir / "test-session.audit.jsonl"
        log_file.write_text('{"event_type":"session_start"}\n')
        with patch("godspeed.audit.trail.AuditTrail") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.log_path.exists.return_value = True
            mock_inst.verify_chain.return_value = (True, "valid chain")
            mock_cls.return_value = mock_inst

            runner = CliRunner()
            audit_args = ["audit", "verify", "test-session", "--audit-dir", str(audit_dir)]
            result = runner.invoke(main, audit_args)
            assert result.exit_code == 0
            assert "VALID" in result.output

    def test_audit_verify_single_session_broken(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        log_file = audit_dir / "test-session.audit.jsonl"
        log_file.write_text('{"event_type":"session_start"}\n')
        with patch("godspeed.audit.trail.AuditTrail") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.log_path.exists.return_value = True
            mock_inst.verify_chain.return_value = (False, "hash mismatch")
            mock_cls.return_value = mock_inst

            runner = CliRunner()
            audit_args = ["audit", "verify", "test-session", "--audit-dir", str(audit_dir)]
            result = runner.invoke(main, audit_args)
            assert result.exit_code != 0
            assert "BROKEN" in result.output

    def test_audit_verify_single_session_not_found(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        with patch("godspeed.audit.trail.AuditTrail") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.log_path.exists.return_value = False
            mock_cls.return_value = mock_inst

            runner = CliRunner()
            audit_args = ["audit", "verify", "test-session", "--audit-dir", str(audit_dir)]
            result = runner.invoke(main, audit_args)
            assert result.exit_code != 0

    def test_models_command_runs(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["models"])
        assert result.exit_code == 0
        assert "Model Presets" in result.output

    def test_ollama_pull_help(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["ollama", "pull", "--help"])
        assert result.exit_code == 0

    def test_doctor_help(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_scan_help(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", "--help"])
        assert result.exit_code == 0


class TestRunApp:
    """Test the _run_app async setup function."""

    def test_resolve_task_input_prompt_file(self, tmp_path: Path) -> None:
        from godspeed.cli import _resolve_task_input

        pfile = tmp_path / "task.txt"
        pfile.write_text("  hello world  ")
        result = _resolve_task_input("", prompt_file=pfile)
        assert result == "hello world"

    def test_resolve_task_input_arg(self) -> None:
        from godspeed.cli import _resolve_task_input

        result = _resolve_task_input("hello", prompt_file=None)
        assert result == "hello"

    def test_resolve_task_input_empty_stdin_tty(self) -> None:
        from godspeed.cli import _resolve_task_input

        with patch("sys.stdin.isatty", return_value=True):
            result = _resolve_task_input("", prompt_file=None)
        assert result == ""

    @patch("godspeed.llm.client.LLMClient")
    @patch("godspeed.agent.conversation.Conversation")
    @patch("godspeed.audit.trail.AuditTrail")
    @patch("godspeed.tools.base.ToolContext")
    @patch("godspeed.security.permissions.PermissionEngine")
    @patch("godspeed.agent.system_prompt.build_system_prompt")
    @patch("godspeed.tui.app.TUIApp")
    @patch("godspeed.tools.tasks.TaskTool")
    @patch("godspeed.tools.tasks.TaskStore")
    @patch("godspeed.context.project_instructions.load_project_instructions")
    @patch("godspeed.context.auto_index.maybe_start_auto_index")
    def test_run_app_creates_audit_dir(
        self,
        mock_auto_index: MagicMock,
        mock_instructions: MagicMock,
        mock_task_store: MagicMock,
        mock_task_tool: MagicMock,
        mock_tui: MagicMock,
        mock_build_prompt: MagicMock,
        mock_perm: MagicMock,
        mock_tool_ctx: MagicMock,
        mock_audit: MagicMock,
        mock_conversation: MagicMock,
        mock_llm: MagicMock,
        tmp_path: Path,
    ) -> None:
        import asyncio

        from godspeed.cli import _run_app

        audit_dir = tmp_path / "audit"
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch("godspeed.config.GodspeedSettings") as mock_settings_cls, \
             patch("godspeed.cli._build_tool_registry") as mock_registry:
            mock_registry.return_value = (MagicMock(), {"shell": "high"})
            s = MagicMock()
            s.model = "test-model"
            s.permissions.deny = []
            s.permissions.allow = []
            s.permissions.ask = []
            s.audit.enabled = True
            s.audit.directory = audit_dir
            s.log_conversations = False
            s.memory_enabled = False
            s.auto_index = False
            s.context.project_instructions = []
            s.fallback_models = []
            s.thinking_budget = 0
            s.max_cost_usd = 0
            s.routing = None
            s.global_dir = tmp_path
            s.code_edit_mode = "edit"
            s.auto_approve = "reads"
            s.competition_mode = False
            s.max_iterations = 50
            s.timeout = 0
            s.platform = "windows"
            mock_settings_cls.return_value = s

            try:
                mock_tui.return_value.run = AsyncMock()
                asyncio.run(
                    _run_app(
                        model="test-model",
                        project_dir=project_dir,
                        verbose=False,
                        audit_dir=audit_dir,
                    )
                )
            except Exception:  # noqa: S110
                pass

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["log_dir"] == audit_dir
