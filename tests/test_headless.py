"""Tests for headless/CI mode."""

from __future__ import annotations

from click.testing import CliRunner

from godspeed.cli import main


class TestHeadlessCommand:
    """Test the `godspeed run` CLI command."""

    def test_run_command_exists(self) -> None:
        """The 'run' subcommand should be registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "headless" in result.output.lower() or "non-interactive" in result.output.lower()

    def test_run_requires_task(self) -> None:
        """Should fail if no task argument provided."""
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_auto_approve_choices(self) -> None:
        """Should accept valid auto-approve levels."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "reads" in result.output
        assert "all" in result.output
        assert "none" in result.output

    def test_json_output_flag(self) -> None:
        """Should have a --json-output flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "json-output" in result.output
