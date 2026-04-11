"""Shell tool — run shell commands via subprocess."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600


def _detect_shell() -> list[str]:
    """Return the shell command prefix for the current platform.

    Uses bash on Unix. On Windows, prefers git-bash, then falls back to cmd.
    """
    if platform.system() != "Windows":
        return ["/bin/bash", "-c"]

    # Prefer git-bash on Windows
    git_bash = shutil.which("bash")
    if git_bash:
        return [git_bash, "-c"]

    # Fallback to cmd
    return ["cmd.exe", "/c"]


class ShellTool(Tool):
    """Run shell commands via subprocess.

    Each invocation is stateless — shell state (cwd changes, env vars) does not
    persist between calls. The working directory is always set from context.cwd.
    Cross-platform: uses bash on Unix, git-bash (or cmd fallback) on Windows.
    """

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Run a shell command and capture stdout/stderr. "
            "Each command runs independently (stateless). "
            "Use absolute paths or paths relative to the project root."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Timeout in seconds (default: {DEFAULT_TIMEOUT}, max: {MAX_TIMEOUT})"
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = arguments.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.failure("command must be a non-empty string")

        raw_timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        if not isinstance(raw_timeout, int):
            try:
                raw_timeout = int(raw_timeout)
            except (TypeError, ValueError):
                return ToolResult.failure(
                    f"timeout must be an integer, got {type(raw_timeout).__name__}"
                )
        if raw_timeout <= 0:
            return ToolResult.failure("timeout must be positive")
        timeout = min(raw_timeout, MAX_TIMEOUT)

        shell_prefix = _detect_shell()
        logger.info("shell.execute command=%r timeout=%d", command, timeout)

        try:
            proc = subprocess.run(
                [*shell_prefix, command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(context.cwd),
            )
        except subprocess.TimeoutExpired:
            logger.warning("shell.timeout command=%r timeout=%d", command, timeout)
            return ToolResult.failure(f"Command timed out after {timeout}s")
        except FileNotFoundError as exc:
            return ToolResult.failure(f"Shell not found: {exc}")

        output_parts: list[str] = []
        if proc.stdout:
            output_parts.append(proc.stdout)
        if proc.stderr:
            output_parts.append(f"STDERR:\n{proc.stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if proc.returncode != 0:
            return ToolResult.failure(f"Exit code {proc.returncode}\n{output}")

        return ToolResult.success(output)
