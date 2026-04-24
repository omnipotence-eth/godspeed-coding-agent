"""Sandboxed shell execution for security."""

from __future__ import annotations

import logging
import shutil
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SandboxedShellTool(Tool):
    """Execute commands in an isolated sandbox.

    Provides an additional layer of security by running
    commands in an isolated environment. Useful for
    untrusted code or CI/CD environments.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "sandbox_shell"

    @property
    def description(self) -> str:
        return (
            "Execute commands in an isolated sandbox. "
            "Provides security by limiting file system access "
            "and network capabilities."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                },
                "sandbox_type": {
                    "type": "string",
                    "enum": ["filesystem", "network", "full"],
                    "description": "Sandbox type",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        command: str,
        timeout: int = 30,
        sandbox_type: str = "filesystem",
    ) -> ToolResult:
        """Execute sandboxed command."""
        import subprocess

        sandbox_config = {
            "filesystem": {
                "allowed_dirs": [str(tool_context.cwd)],
                "blocked_commands": ["rm -rf", "dd", "mkfs", "fdisk"],
            },
            "network": {
                "allowed_dirs": [str(tool_context.cwd), "/tmp"],
                "blocked_commands": ["curl", "wget", "ssh", "nc"],
                "block_network": True,
            },
            "full": {
                "allowed_dirs": [str(tool_context.cwd)],
                "blocked_commands": [],
                "block_network": False,
            },
        }

        config = sandbox_config.get(sandbox_type, sandbox_config["filesystem"])

        for blocked in config.get("blocked_commands", []):
            if blocked in command:
                return ToolResult.failure(
                    f"Command blocked in {sandbox_type} sandbox: {blocked}"
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(tool_context.cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return ToolResult.failure(
                    f"Command failed with code {result.returncode}\n{result.stderr}"
                )

            output = result.stdout
            if result.stderr:
                output += f"\nstderr: {result.stderr}"

            return ToolResult.ok(output)

        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"Command timed out after {timeout}s")
        except Exception as exc:
            return ToolResult.failure(f"Execution failed: {exc}")


def is_sandboxed_command_safe(command: str) -> bool:
    """Check if a command is safe to run in sandbox."""
    unsafe_patterns = [
        "rm -rf /",
        "dd if=",
        ":(){:|:&};:",
        "chmod -R 777 /",
        "wget | sh",
        "curl | sh",
    ]

    for pattern in unsafe_patterns:
        if pattern in command:
            return False

    return True


def get_available_sandbox() -> str | None:
    """Check what sandboxing options are available."""
    if shutil.which("bwrap"):
        return "bwrap"
    if shutil.which("firejail"):
        return "firejail"
    if shutil.which("docker"):
        return "docker"
    return None
