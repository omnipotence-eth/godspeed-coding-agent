"""Tests for shell tool."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.shell import ShellTool


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


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
        mock_result = MagicMock()
        mock_result.stdout = "hello world\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("godspeed.tools.shell.subprocess.run", return_value=mock_result):
            result = await tool.execute({"command": "echo hello world"}, tool_context)

        assert not result.is_error
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_command_with_stderr(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "output\n"
        mock_result.stderr = "warning: something\n"
        mock_result.returncode = 0

        with patch("godspeed.tools.shell.subprocess.run", return_value=mock_result):
            result = await tool.execute({"command": "some-cmd"}, tool_context)

        assert not result.is_error
        assert "output" in result.output
        assert "STDERR" in result.output
        assert "warning" in result.output

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, tool: ShellTool, tool_context: ToolContext) -> None:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error: not found\n"
        mock_result.returncode = 1

        with patch("godspeed.tools.shell.subprocess.run", return_value=mock_result):
            result = await tool.execute({"command": "bad-cmd"}, tool_context)

        assert result.is_error
        assert "Exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self, tool: ShellTool, tool_context: ToolContext) -> None:
        with patch(
            "godspeed.tools.shell.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="sleep 999", timeout=5),
        ):
            result = await tool.execute({"command": "sleep 999", "timeout": 5}, tool_context)

        assert result.is_error
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_clamped_to_max(self, tool: ShellTool, tool_context: ToolContext) -> None:
        """Timeout exceeding MAX_TIMEOUT is clamped."""
        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("godspeed.tools.shell.subprocess.run", return_value=mock_result) as mock_run:
            await tool.execute({"command": "echo ok", "timeout": 9999}, tool_context)

        # The actual timeout passed to subprocess should be MAX_TIMEOUT (600)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["timeout"] == 600

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
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("godspeed.tools.shell.subprocess.run", return_value=mock_result):
            result = await tool.execute({"command": "true"}, tool_context)

        assert not result.is_error
        assert "no output" in result.output.lower()
