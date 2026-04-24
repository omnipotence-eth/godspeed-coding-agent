"""Hooks system for lifecycle automation."""

from __future__ import annotations

import logging
from typing import Any
from enum import Enum

logger = logging.getLogger(__name__)


class HookType(str, Enum):
    """Types of hooks."""

    PRE_TOOL = "PreToolUse"
    POST_TOOL = "PostToolUse"
    NOTIFICATION = "Notification"
    STOP = "Stop"


_HOOKS: dict[str, list[callable]] = {
    HookType.PRE_TOOL: [],
    HookType.POST_TOOL: [],
    HookType.NOTIFICATION: [],
    HookType.STOP: [],
}


def register_hook(hook_type: HookType, handler: callable) -> None:
    """Register a hook handler."""
    if hook_type not in _HOOKS:
        _HOOKS[hook_type] = []
    _HOOKS[hook_type].append(handler)
    logger.info("Registered hook: %s", hook_type)


def unregister_hook(hook_type: HookType, handler: callable) -> None:
    """Unregister a hook handler."""
    if hook_type in _HOOKS and handler in _HOOKS[hook_type]:
        _HOOKS[hook_type].remove(handler)


async def run_pre_tool_hooks(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run PreToolUse hooks. Returns modified args or None to block."""
    for hook in _HOOKS.get(HookType.PRE_TOOL, []):
        try:
            result = await hook(tool_name, args)
            if result is not None:
                args = result
        except Exception as exc:
            logger.warning("PreToolUse hook error: %s", exc)
    return args


async def run_post_tool_hooks(
    tool_name: str, args: dict[str, Any], result: Any
) -> Any:
    """Run PostToolUse hooks. Can modify result."""
    for hook in _HOOKS.get(HookType.POST_TOOL, []):
        try:
            new_result = await hook(tool_name, args, result)
            if new_result is not None:
                result = new_result
        except Exception as exc:
            logger.warning("PostToolUse hook error: %s", exc)
    return result


async def run_stop_hook(response: str) -> None:
    """Run Stop hooks."""
    for hook in _HOOKS.get(HookType.STOP, []):
        try:
            await hook(response)
        except Exception as exc:
            logger.warning("Stop hook error: %s", exc)


class HookTool(Tool):
    """Manage hooks for the agent."""

    name = "hook_manage"
    description = "Register and manage lifecycle hooks"
    risk_level = RiskLevel.LOW

    async def execute(self, tool_context, action, hook_type=None, handler_code=None):
        """Manage hooks."""
        from godspeed.tools.base import ToolResult

        if action == "list":
            hooks = []
            for ht, handlers in _HOOKS.items():
                for h in handlers:
                    hooks.append(f"{ht}: {h.__name__ if hasattr(h, '__name__') else str(h)}")
            return ToolResult.ok("\n".join(hooks) if hooks else "No hooks registered")

        return ToolResult.failure(f"Unknown action: {action}")


from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult