"""Hook executor — runs shell commands at agent lifecycle events."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from godspeed.hooks.config import HookDefinition

logger = logging.getLogger(__name__)


class HookExecutor:
    """Execute hooks at agent lifecycle events.

    Pre-tool hooks can block tool execution by returning a non-zero exit code.
    All hooks run synchronously in a subprocess with a timeout.

    Args:
        hooks: List of hook definitions to execute.
        cwd: Working directory for subprocess execution.
        session_id: Current session ID for template expansion.
    """

    def __init__(
        self,
        hooks: list[HookDefinition],
        cwd: Path,
        session_id: str,
    ) -> None:
        self._hooks = hooks
        self._cwd = cwd
        self._session_id = session_id

    def run_pre_tool(self, tool_name: str) -> bool:
        """Run pre_tool_call hooks. Returns True to proceed, False to abort."""
        for hook in self._hooks:
            if hook.event != "pre_tool_call":
                continue
            if not self._matches_tool(hook, tool_name):
                continue

            exit_code = self._execute(hook, {"tool_name": tool_name})
            if exit_code != 0:
                logger.warning(
                    "Pre-tool hook blocked tool=%s command=%s exit=%d",
                    tool_name,
                    hook.command,
                    exit_code,
                )
                return False
        return True

    def run_post_tool(self, tool_name: str, result: str = "") -> None:
        """Run post_tool_call hooks."""
        for hook in self._hooks:
            if hook.event != "post_tool_call":
                continue
            if not self._matches_tool(hook, tool_name):
                continue

            self._execute(hook, {"tool_name": tool_name})

    def run_pre_session(self) -> None:
        """Run pre_session hooks."""
        for hook in self._hooks:
            if hook.event == "pre_session":
                self._execute(hook, {})

    def run_post_session(self) -> None:
        """Run post_session hooks."""
        for hook in self._hooks:
            if hook.event == "post_session":
                self._execute(hook, {})

    def _matches_tool(self, hook: HookDefinition, tool_name: str) -> bool:
        """Check if a hook applies to the given tool."""
        return hook.tools is None or tool_name in hook.tools

    def _execute(self, hook: HookDefinition, context: dict[str, str]) -> int:
        """Execute a hook command with template variable expansion.

        Returns the exit code (0 = success).
        """
        template_vars = {
            "session_id": self._session_id,
            "cwd": str(self._cwd),
            "project_dir": str(self._cwd),
            **context,
        }

        try:
            command = hook.command.format(**template_vars)
        except KeyError as exc:
            logger.warning(
                "Hook template error command=%s missing_var=%s",
                hook.command,
                exc,
            )
            return 1

        logger.debug(
            "Executing hook event=%s command=%s timeout=%d",
            hook.event,
            command,
            hook.timeout,
        )

        try:
            result = subprocess.run(  # noqa: S602  # nosec B602
                command,
                shell=True,
                cwd=self._cwd,
                timeout=hook.timeout,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                logger.debug("Hook stdout: %s", result.stdout.strip())
            if result.stderr:
                logger.debug("Hook stderr: %s", result.stderr.strip())
            return result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(
                "Hook timed out event=%s command=%s timeout=%ds",
                hook.event,
                command,
                hook.timeout,
            )
            return 1
        except OSError as exc:
            logger.warning(
                "Hook execution failed event=%s command=%s error=%s",
                hook.event,
                command,
                exc,
            )
            return 1
