"""Tests for the headless `godspeed run` command (v2.5.0).

Covers the MLOps-pipeline-facing contract: task input resolution,
exit-code mapping, JSON output schema, --timeout wiring, and audit
trail auto-creation.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.agent.result import ExitCode
from godspeed.cli import _resolve_task_input


class TestResolveTaskInput:
    """_resolve_task_input precedence: --prompt-file > TASK > stdin."""

    def test_prompt_file_wins_over_positional(self, tmp_path: Path) -> None:
        prompt = tmp_path / "task.md"
        prompt.write_text("task from file", encoding="utf-8")
        assert _resolve_task_input("ignored", prompt) == "task from file"

    def test_positional_task_used_when_no_file(self) -> None:
        assert _resolve_task_input("do the thing", None) == "do the thing"

    def test_dash_reads_from_stdin(self) -> None:
        with patch("sys.stdin", StringIO("piped task\n")):
            assert _resolve_task_input("-", None) == "piped task"

    def test_empty_task_and_pipe_stdin_reads_stdin(self) -> None:
        fake_stdin = StringIO("from pipe\n")
        # isatty defaults to False on StringIO → resolve treats it as a pipe.
        with patch("sys.stdin", fake_stdin):
            assert _resolve_task_input("", None) == "from pipe"

    def test_empty_task_and_tty_stdin_returns_empty(self) -> None:
        """Interactive shell with no positional and no pipe → INVALID_INPUT."""

        class _TTYStream:
            def isatty(self) -> bool:
                return True

            def read(self) -> str:  # pragma: no cover — never called
                return "should not be read"

        with patch("sys.stdin", _TTYStream()):
            assert _resolve_task_input("", None) == ""

    def test_prompt_file_strips_trailing_whitespace(self, tmp_path: Path) -> None:
        prompt = tmp_path / "task.md"
        prompt.write_text("  task  \n\n", encoding="utf-8")
        assert _resolve_task_input("", prompt) == "task"


class TestExitCodeOrdering:
    """The exit-code contract is public API. Values must not drift."""

    def test_success_is_zero(self) -> None:
        assert int(ExitCode.SUCCESS) == 0

    def test_exit_codes_are_stable(self) -> None:
        """If this test fails, a minor-version bump has broken pipeline
        orchestrators that switch on exit code. Bump the major version."""
        assert int(ExitCode.SUCCESS) == 0
        assert int(ExitCode.TOOL_ERROR) == 1
        assert int(ExitCode.MAX_ITERATIONS) == 2
        assert int(ExitCode.BUDGET_EXCEEDED) == 3
        assert int(ExitCode.LLM_ERROR) == 4
        assert int(ExitCode.INVALID_INPUT) == 5
        assert int(ExitCode.TIMEOUT) == 6
        assert int(ExitCode.INTERRUPTED) == 130


class TestHeadlessRunCommand:
    """Integration-style smoke test for the Click command wiring.

    We don't exercise the full agent loop (would need a live model) — we
    verify the flag parsing, input resolution, and error paths.
    """

    def test_invalid_input_exit_code(self) -> None:
        """No task and no stdin pipe → exit 5 (INVALID_INPUT)."""
        from click.testing import CliRunner

        from godspeed.cli import headless_run

        runner = CliRunner()

        class _TTYStream:
            def isatty(self) -> bool:
                return True

            def read(self) -> str:
                return ""

        with patch("sys.stdin", _TTYStream()):
            result = runner.invoke(headless_run, [], standalone_mode=False)

        assert result.exit_code == int(ExitCode.INVALID_INPUT)

    def test_timeout_flag_accepted(self) -> None:
        """--timeout N parses without error (won't actually run to timeout)."""
        from click.testing import CliRunner

        from godspeed.cli import headless_run

        runner = CliRunner()
        result = runner.invoke(
            headless_run,
            ["--timeout", "30", "--help"],
            standalone_mode=False,
        )
        # --help short-circuits Click; we just confirm the flag is valid.
        assert result.exit_code == 0
        assert "--timeout" in result.output

    def test_prompt_file_flag_registered(self) -> None:
        from click.testing import CliRunner

        from godspeed.cli import headless_run

        runner = CliRunner()
        result = runner.invoke(headless_run, ["--help"], standalone_mode=False)
        assert "--prompt-file" in result.output


class TestEnrichedJsonSchema:
    """The JSON output contract is documented in the CLI help and consumed
    by pipeline orchestrators. Every field must be present when --json-output
    is set, even when the run is a no-op.

    We assert the schema shape by inspecting `_headless_run`'s local output
    dict construction — the true integration test lives in the e2e suite.
    """

    def test_expected_fields_are_referenced(self) -> None:
        """All promised JSON fields appear in the cli.py source."""
        import inspect

        from godspeed import cli

        source = inspect.getsource(cli._headless_run)
        for field in [
            '"task"',
            '"model"',
            '"session_id"',
            '"response"',
            '"exit_reason"',
            '"exit_code"',
            '"iterations_used"',
            '"tool_calls"',
            '"tool_call_count"',
            '"tool_error_count"',
            '"duration_seconds"',
            '"input_tokens"',
            '"output_tokens"',
            '"cost_usd"',
            '"audit_log_path"',
        ]:
            assert field in source, f"Missing JSON field reference: {field}"


@pytest.mark.asyncio
class TestHeadlessRunIntegration:
    """End-to-end test with a mocked LLM client — verifies exit codes
    and audit trail creation without needing a live API."""

    async def test_success_path_creates_audit_and_returns_zero(
        self, tmp_path: Path, monkeypatch
    ) -> None:

        from godspeed.cli import _headless_run
        from godspeed.llm.client import ChatResponse

        # Redirect global_dir to tmp_path so audit writes to a test location.
        monkeypatch.setenv("GODSPEED_GLOBAL_DIR", str(tmp_path / "godspeed"))

        # Mock LLMClient.chat to return a clean final text response.
        async def fake_chat(self, messages, tools=None, task_type=None):
            return ChatResponse(content="hello world", finish_reason="stop")

        with (
            patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
            patch("sys.stdout.write"),
        ):
            exit_code = await _headless_run(
                task="say hello",
                model="test-model",
                project_dir=tmp_path,
                auto_approve="reads",
                max_iterations=5,
                timeout=0,
                json_output=False,
            )

        assert exit_code == int(ExitCode.SUCCESS)
        # Audit directory was created.
        audit_dir = tmp_path / "godspeed" / "audit"
        assert audit_dir.exists()
        audit_files = list(audit_dir.glob("*.audit.jsonl"))
        assert len(audit_files) == 1
