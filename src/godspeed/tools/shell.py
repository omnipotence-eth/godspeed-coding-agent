"""Shell tool — run shell commands via subprocess."""

from __future__ import annotations

import contextlib
import logging
import platform
import shutil
import subprocess
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_COMMAND_LENGTH = 10000  # 10K characters max for shell commands
MAX_COMMAND_LENGTH = 10000  # 10K characters max for shell commands


def _kill_process_tree(pid: int) -> None:
    """Force-kill a process and all its descendants.

    Why this exists:
      ``subprocess.run(..., timeout=N)`` is documented to kill the child
      on TimeoutExpired, but on Windows (and sometimes on Linux with
      certain pipe configurations) that kill does NOT propagate to
      grandchildren. When the agent runs ``shell(command='python')`` the
      shell spawns git-bash which spawns an interactive Python — killing
      git-bash leaves Python holding stdout/stderr pipes, and
      subprocess.run blocks indefinitely waiting for them to close.

      Observed in SWE-Bench dev-23 attempt #3: instance sqlfluff-1517
      hung for ~100 minutes after a bare ``python`` REPL call despite
      the tool's 120s timeout. Instance sqlfluff-1733 hung ~60 min on a
      recursive ``sqlfluff fix``. Both required manual PID kill to
      unstick.

    This helper uses psutil's ``children(recursive=True)`` to walk the
    tree and issue kill() to each — which translates to
    ``TerminateProcess`` on Windows and SIGKILL on Unix. Cross-platform.

    Best-effort: if any process in the tree has already exited we skip
    it silently. Never raises to the caller.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available; cannot force-kill process tree for pid=%d", pid)
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            child.kill()
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        parent.kill()


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
            "Use absolute paths or paths relative to the project root. "
            "Set background=true for long-running commands, then use "
            "background_check to poll status.\n\n"
            "Example: shell(command='pytest tests/ -v')\n"
            "Example: shell(command='pip install requests', timeout=60)\n"
            "Example: shell(command='npm run build', background=true)"
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
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run in background and return immediately. "
                        "Use background_check tool to poll status."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = arguments.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.failure("command must be a non-empty string")

        # Check command length limit
        if len(command) > MAX_COMMAND_LENGTH:
            return ToolResult.failure(
                f"Command exceeds maximum length of {MAX_COMMAND_LENGTH} characters"
            )

        # Check command length limit
        if len(command) > MAX_COMMAND_LENGTH:
            return ToolResult.failure(
                f"Command exceeds maximum length of {MAX_COMMAND_LENGTH} characters"
            )

        # Background execution
        if arguments.get("background", False):
            return await self._execute_background(command, context)

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

        # Use Popen + communicate(timeout=...) instead of subprocess.run so
        # we can explicitly kill the process tree on timeout. subprocess.run's
        # timeout cleanup is unreliable on Windows when the child has holding
        # pipes (see _kill_process_tree docstring).
        try:
            proc = subprocess.Popen(
                [*shell_prefix, command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(context.cwd),
            )
        except FileNotFoundError as exc:
            return ToolResult.failure(f"Shell not found: {exc}")

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            logger.warning(
                "shell.timeout pid=%d command=%r timeout=%d - force-killing process tree",
                proc.pid,
                command,
                timeout,
            )
            _kill_process_tree(proc.pid)
            # After killing the tree, drain any buffered output so the
            # underlying pipe FDs close and we don't leak them. Give it
            # a short window; if still blocked, move on with empty output.
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            tail = ""
            if stdout:
                tail += f"\nSTDOUT tail:\n{stdout[-2000:]}"
            if stderr:
                tail += f"\nSTDERR tail:\n{stderr[-2000:]}"
            return ToolResult.failure(
                f"Command timed out after {timeout}s and was force-killed "
                f"(including any child processes).{tail}"
            )

        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if returncode != 0:
            return ToolResult.failure(f"Exit code {returncode}\n{output}")

        return ToolResult.success(output)

    async def _execute_background(self, command: str, context: ToolContext) -> ToolResult:
        """Spawn a command in the background and return its process ID."""
        import asyncio
        import time

        from godspeed.tools.background import (
            MAX_CONCURRENT,
            BackgroundProcess,
            BackgroundRegistry,
            _collect_output,
        )

        registry = BackgroundRegistry.get()

        if registry.active_count >= MAX_CONCURRENT:
            return ToolResult.failure(
                f"Too many background processes ({registry.active_count}/{MAX_CONCURRENT}). "
                "Kill some before starting new ones."
            )

        shell_prefix = _detect_shell()
        logger.info("shell.background command=%r", command)

        proc = await asyncio.create_subprocess_exec(
            *shell_prefix,
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(context.cwd),
        )

        pid = registry.next_id()
        bg_proc = BackgroundProcess(
            id=pid,
            command=command,
            process=proc,
            started_at=time.monotonic(),
        )
        # Start collecting output in background
        bg_proc._collection_task = asyncio.create_task(_collect_output(bg_proc))
        registry.add(bg_proc)

        return ToolResult.success(
            f"Started background process {pid}\n"
            f"Command: {command}\n"
            f"Use background_check to poll status."
        )
