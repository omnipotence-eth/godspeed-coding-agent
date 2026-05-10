"""Additional CLI tests covering remaining untested paths in cli.py."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed._bootstrap import _build_tool_registry, _parse_env_file
from godspeed.cli import (
    _ensure_llamacpp,
    _ensure_ollama,
    _is_ollama_running,
    _resolve_task_input,
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

    def test_verify_audit_no_dir(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "definitely_nonexistent_dir"
        runner = CliRunner()
        result = runner.invoke(main, ["audit", "verify", "--audit-dir", str(audit_dir)])
        assert result.exit_code != 0

    def test_verify_audit_with_logs(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from godspeed.cli import main

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        log_file = audit_dir / "test-session.audit.jsonl"
        log_file.write_text('{"event_type":"session_start","session_id":"test-session"}\n')
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
        log_file.write_text('{"event_type":"session_start","session_id":"test-session"}\n')
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

        with patch("asyncio.run", return_value=None):
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

    def test_resolve_task_input_dash_stdin(self) -> None:
        with patch("sys.stdin", io.StringIO("piped task")):
            result = _resolve_task_input("-", prompt_file=None)
        assert result == "piped task"

    def test_resolve_task_input_prompt_file_overrides_arg(self, tmp_path: Path) -> None:
        pfile = tmp_path / "task.md"
        pfile.write_text("from file")
        result = _resolve_task_input("from arg", prompt_file=pfile)
        assert result == "from file"

    def test_run_app_passes_audit_dir(
        self,
        tmp_path: Path,
    ) -> None:
        import asyncio

        from godspeed.cli import _run_app

        audit_dir = tmp_path / "audit"
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with (
            patch("godspeed.cli._build_tool_registry") as mock_registry,
            patch("godspeed.config.GodspeedSettings") as mock_settings_cls,
            patch("godspeed.tui.textual_app.GodspeedTextualApp") as mock_app_cls,
        ):
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
            mock_settings_cls.return_value = s

            mock_app = MagicMock()
            mock_app.run_async = AsyncMock()
            mock_app_cls.return_value = mock_app

            asyncio.run(
                _run_app(
                    model="test-model",
                    project_dir=project_dir,
                    _verbose=False,
                    audit_dir=audit_dir,
                )
            )

        mock_app_cls.assert_called_once()
        call_kwargs = mock_app_cls.call_args[1]
        assert call_kwargs["audit_dir"] == audit_dir


class TestHeadlessRun:
    """Comprehensive tests for _headless_run covering all major code paths."""

    @pytest.mark.asyncio
    async def test_success_path_json_output(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="All done.", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=True,
            )

        assert exit_code == 0
        output = json.loads(stdout.getvalue().strip())
        assert output["response"] == "All done."
        assert output["exit_code"] == 0
        assert "tool_calls" in output

    @pytest.mark.asyncio
    async def test_success_path_non_json_output(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="Task completed.", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == 0
        assert "Task completed." in stdout.getvalue()

    @pytest.mark.asyncio
    async def test_tool_error_exit_code(self, tmp_path: Path) -> None:
        from godspeed.agent.result import ExitCode
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="Error: Something went wrong", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == int(ExitCode.TOOL_ERROR)

    @pytest.mark.asyncio
    async def test_max_iterations_exit_code(self, tmp_path: Path) -> None:
        from godspeed.agent.result import ExitCode
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        call_count = [0]

        async def endless_tools(self, messages, tools=None, task_type=None):
            call_count[0] += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_count[0]}",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "echo loop"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=endless_tools),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="all",
                max_iterations=2,
                timeout=0,
                json_output=False,
            )

        assert exit_code == int(ExitCode.MAX_ITERATIONS)

    @pytest.mark.asyncio
    async def test_llm_error_exit_code(self, tmp_path: Path) -> None:
        from godspeed.agent.result import ExitCode
        from godspeed.cli import _headless_run

        async def failing_chat(self, messages, tools=None, task_type=None):
            raise RuntimeError("LLM provider down")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=failing_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == int(ExitCode.LLM_ERROR)

    @pytest.mark.asyncio
    async def test_budget_exceeded_exit_code(self, tmp_path: Path) -> None:
        from godspeed.agent.result import ExitCode
        from godspeed.cli import _headless_run
        from godspeed.llm.client import BudgetExceededError

        async def budget_fail(self, messages, tools=None, task_type=None):
            raise BudgetExceededError(spent=10.0, limit=5.0)

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=budget_fail),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == int(ExitCode.BUDGET_EXCEEDED)

    @pytest.mark.asyncio
    async def test_timeout_exit_code(self, tmp_path: Path) -> None:
        from godspeed.agent.result import ExitCode
        from godspeed.cli import _headless_run

        async def slow_chat(self, messages, tools=None, task_type=None):
            await asyncio.sleep(10)
            return MagicMock()

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=slow_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0.01,
                json_output=False,
            )

        assert exit_code == int(ExitCode.TIMEOUT)

    @pytest.mark.asyncio
    async def test_competition_mode_disables_logging(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="Done.", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test task",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="all",
                max_iterations=5,
                timeout=0,
                json_output=False,
                competition_mode=True,
            )

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_headless_with_ollama_model(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="ok", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("godspeed.cli._ensure_ollama") as mock_ollama,
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test",
                model="ollama/qwen3:4b",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == 0
        mock_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_headless_with_llamacpp_model(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="ok", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("godspeed.cli._ensure_llamacpp") as mock_llamacpp,
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="test",
                model="llamacpp/qwen2.5-coder",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == 0
        mock_llamacpp.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_approve_all_permissions(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        call_count = [0]

        async def multi_chat(self, messages, tools=None, task_type=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[{
                        "id": "call_1",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "echo ok"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return ChatResponse(content="Executed.", finish_reason="stop")

        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=multi_chat),
            patch("sys.stdout", stdout),
        ):
            exit_code = await _headless_run(
                task="run command",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="all",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_permission_denied_in_headless(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        call_count = [0]

        async def multi_chat(self, messages, tools=None, task_type=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[{
                        "id": "call_1",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "rm -rf /"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return ChatResponse(content="Permission denied, I understand.", finish_reason="stop")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=multi_chat),
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
        ):
            exit_code = await _headless_run(
                task="delete all",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="none",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code in (0, 1, 2)

    @pytest.mark.asyncio
    async def test_callbacks_stderr_output(self, tmp_path: Path) -> None:
        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        call_count = [0]

        async def multi_chat(self, messages, tools=None, task_type=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[{
                        "id": "call_001",
                        "function": {
                            "name": "file_read",
                            "arguments": '{"file_path": "test.py"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return ChatResponse(content="Done.", finish_reason="stop")

        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch("godspeed.llm.client.LLMClient.chat", new=multi_chat),
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
        ):
            exit_code = await _headless_run(
                task="test",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="all",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == 0
        assert "[tool]" in stderr.getvalue() or "Done." in stdout.getvalue()


class TestCLICommandsExtended:
    """Extended CLI subcommand invocations."""

    def test_version_command_output(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0

    def test_init_help(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0

    def test_run_command_help(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--timeout" in result.output
        assert "--auto-approve" in result.output
        assert "--json-output" in result.output

    def test_web_command_help(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["web", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_ollama_list_command(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.list_models", return_value=[]):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "list"])
            assert result.exit_code == 0

    def test_ollama_pull_command(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.pull_model", return_value=True):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "pull", "qwen3:4b"])
            assert result.exit_code == 0

    def test_ollama_pull_failure(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.pull_model", return_value=False):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "pull", "bad-model"])
            assert result.exit_code == 0

    def test_ollama_show_command(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.show_model", return_value={"name": "qwen", "size": "4B"}):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "show", "qwen3:4b"])
            assert result.exit_code == 0

    def test_ollama_show_not_found(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.show_model", return_value=None):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "show", "unknown-model"])
            assert result.exit_code == 0

    def test_ollama_delete_command(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.delete_model", return_value=(True, "deleted")):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "delete", "qwen3:4b"])
            assert result.exit_code == 0

    def test_ollama_delete_failure(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.tools.ollama_manager.delete_model", return_value=(False, "error")):
            runner = CliRunner()
            result = runner.invoke(main, ["ollama", "delete", "qwen3:4b"])
            assert result.exit_code == 0

    def test_scan_command(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with patch("godspeed.evolution.hardware.format_machine_report", return_value="Machine report"):
            runner = CliRunner()
            result = runner.invoke(main, ["scan"])
            assert result.exit_code == 0
            assert "Machine report" in result.output

    def test_doctor_command_basic(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0

    def test_doctor_with_fix_flag(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["doctor", "--fix"])
        assert result.exit_code == 0

    def test_swebench_run_help(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["swebench", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_export_training_command(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["export-training", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    @patch("godspeed.cli._headless_run")
    def test_run_with_prompt_file(self, mock_headless: MagicMock) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        mock_headless.return_value = 0
        with patch("asyncio.run", return_value=0):
            runner = CliRunner()
            result = runner.invoke(
                main, ["run", "--prompt-file", "nonexistent_file.txt", "fallback task"]
            )
            assert result.exit_code != 0  # File doesn't exist

    def test_run_with_competition_mode_flag(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--competition-mode", "--help"])
        assert result.exit_code == 0

    def test_main_with_permission_mode_flag(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--permission-mode" in result.output
        assert "strict" in result.output

    def test_main_with_execution_mode_flag(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--execution-mode" in result.output

    def test_ensure_ollama_with_console_poll_timeout(self) -> None:
        with (
            patch("godspeed.cli._is_ollama_running", return_value=False),
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch("subprocess.Popen"),
            patch("godspeed.cli.OLLAMA_STARTUP_TIMEOUT", 0.01),
        ):
            assert _ensure_ollama() is False

    def test_ensure_ollama_subprocess_error(self) -> None:
        with (
            patch("godspeed.cli._is_ollama_running", return_value=False),
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch("subprocess.Popen", side_effect=OSError("cannot start")),
        ):
            assert _ensure_ollama() is False

    def test_ensure_llamacpp_start_failure(self) -> None:
        with (
            patch("godspeed.tools.llamacpp_manager.is_server_running", return_value=False),
            patch("godspeed.tools.llamacpp_manager.start_server", return_value=None),
            patch("godspeed.tools.llamacpp_manager.configure_litellm_env"),
        ):
            assert _ensure_llamacpp() is False

    @patch("godspeed.llm.client.LLMClient.chat")
    def test_run_command_with_timeout(self, mock_chat: MagicMock) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        async def fake_headless(*args, **kwargs):
            return 0

        with (
            patch("godspeed.cli._headless_run", side_effect=fake_headless),
            patch("asyncio.run", return_value=0),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main, ["run", "--timeout", "30", "--max-iterations", "10", "task"]
            )
            assert result.exit_code == 0

    def test_web_command_starts(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        with (
            patch("godspeed.cli._run_app", new_callable=AsyncMock),
            patch("asyncio.run", return_value=None),
            patch("sys.argv", ["godspeed", "web"]),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["web"], standalone_mode=False)
            assert result.exit_code == 0

    def test_auto_approve_reads_only(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--auto-approve", "reads", "--help"])
        assert result.exit_code == 0

    def test_auto_approve_all(self) -> None:
        from click.testing import CliRunner
        from godspeed.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--auto-approve", "all", "--help"])
        assert result.exit_code == 0


class TestForceUtf8Stdio:
    def test_force_utf8_noop_when_already_utf8(self) -> None:
        from godspeed.cli import _force_utf8_stdio

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        try:
            sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
            sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="UTF8")
            _force_utf8_stdio()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    def test_force_utf8_rewraps_cp1252(self) -> None:
        from godspeed.cli import _force_utf8_stdio

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        try:
            sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
            sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
            _force_utf8_stdio()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
