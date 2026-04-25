"""Tests for shell tool.

After commit 38e7a9d (force-kill process tree fix), ShellTool uses
``subprocess.Popen`` + ``.communicate(timeout=...)`` instead of
``subprocess.run``. Mocks here patch Popen accordingly.

Live timeout / force-kill behavior is covered in tests/test_shell_tool.py
(spawns real subprocesses); this file uses mocks for fast unit-level
coverage of stdout/stderr handling, exit-code dispatch, timeout error
shape, and timeout clamping to MAX_TIMEOUT.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.shell import ShellTool


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


def _mock_popen(*, stdout: str, stderr: str, returncode: int) -> MagicMock:
    """Build a MagicMock that quacks like a subprocess.Popen object.

    ``communicate(timeout=...)`` returns ``(stdout, stderr)`` and the
    mock's ``returncode`` attribute is set so callers reading
    ``proc.returncode`` after communicate() see the expected value.
    """
    mock = MagicMock()
    mock.communicate.return_value = (stdout, stderr)
    mock.returncode = returncode
    mock.pid = 12345
    return mock


class TestShellTool:
    """Test shell command execution."""

    def test_metadata(self, tool: ShellTool) -> None:
        assert tool.name == "shell"
        assert tool.risk_level == "high"

    def test_schema_has_required_command(self, tool: ShellTool) -> None:
        schema = tool.get_schema()
        assert "command" in schema["properties"]
        assert schema["required"] == ["command"]

    @pytest.mark.asyncio
    async def test_successful_command(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="hello world\n", stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "echo hello world"}, tool_context)

        assert not result.is_error
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_command_with_stderr(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """A command exiting 0 with stderr warnings is still success; stderr is surfaced."""
        mock_proc = _mock_popen(stdout="output\n", stderr="warning: something\n", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "some-cmd"}, tool_context)

        assert not result.is_error
        assert "output" in result.output
        assert "STDERR" in result.output
        assert "warning" in result.output

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="", stderr="error: not found\n", returncode=1)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "bad-cmd"}, tool_context)

        assert result.is_error
        assert "Exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """TimeoutExpired from communicate() must surface as a timeout failure."""
        mock_proc = MagicMock()
        # First communicate(timeout=...) raises; the post-kill drain returns empty
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="sleep 999", timeout=5),
            ("", ""),
        ]
        mock_proc.pid = 99999

        with (
            patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc),
            patch("godspeed.tools.shell._kill_process_tree") as mock_kill,
        ):
            result = await tool.execute({"command": "sleep 999", "timeout": 5}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()
        # Force-kill must be triggered with the proc's pid.
        mock_kill.assert_called_once_with(99999)

    @pytest.mark.asyncio
    async def test_timeout_clamped_to_max(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """Timeout exceeding MAX_TIMEOUT is clamped before being passed to communicate()."""
        mock_proc = _mock_popen(stdout="ok", stderr="", returncode=0)

        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            await tool.execute({"command": "echo ok", "timeout": 9999}, tool_context)

        # communicate(timeout=...) should be called with the clamped value (600).
        mock_proc.communicate.assert_called_once()
        assert mock_proc.communicate.call_args.kwargs["timeout"] == 600

    @pytest.mark.asyncio
    async def test_empty_command(self, tool: ShellTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"command": ""}, tool_context)
        assert result.is_error
        assert "non-empty string" in result.error.lower()

    @pytest.mark.asyncio
    async def test_negative_timeout(self, tool: ShellTool, tool_context: ToolContext) -> None:
        result = await tool.execute({"command": "echo hi", "timeout": -1}, tool_context)
        assert result.is_error
        assert "positive" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_output(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_proc = _mock_popen(stdout="", stderr="", returncode=0)
        with patch("godspeed.tools.shell.subprocess.Popen", return_value=mock_proc):
            result = await tool.execute({"command": "true"}, tool_context)

        assert not result.is_error
        assert "no output" in result.output.lower()

    @pytest.mark.asyncio
    async def test_command_length_limit(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """Test that commands exceeding MAX_COMMAND_LENGTH are rejected."""
        long_command = "echo " + "x" * 10001  # Exceeds 10000 char limit
        result = await tool.execute({"command": long_command}, tool_context)
        assert result.is_error
        assert "exceeds maximum length" in result.error.lower()
