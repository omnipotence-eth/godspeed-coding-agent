"""Flow awareness - tracks user actions in real-time for context."""

from __future__ import annotations

import logging
import time
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


class FlowAction:
    """An action tracked by flow awareness."""

    def __init__(
        self,
        action_type: str,
        details: dict[str, Any],
        timestamp: float | None = None,
    ) -> None:
        self.action_type = action_type
        self.details = details
        self.timestamp = timestamp or time.time()


class FlowAwareness:
    """Tracks real-time user actions for context awareness.

    Similar to Windsurf's Flow awareness that tracks your
    actions in real-time to infer intent.
    """

    MAX_HISTORY = 50

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.actions: deque[FlowAction] = deque(maxlen=self.MAX_HISTORY)
        self.current_task: str | None = None

    def record_action(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a user action."""
        if not self.enabled:
            return

        action = FlowAction(action_type, details or {})
        self.actions.append(action)
        logger.debug("FlowAction: %s", action_type)

    def record_file_edit(self, file_path: str, old_content: str, new_content: str) -> None:
        """Record a file edit action."""
        self.record_action("file_edit", {
            "file_path": file_path,
            "lines_changed": len(new_content.split("\n")) - len(old_content.split("\n")),
        })

    def record_terminal(self, command: str, cwd: str) -> None:
        """Record a terminal command."""
        self.record_action("terminal", {
            "command": command.split()[0] if command else "",
            "full_command": command,
            "cwd": cwd,
        })

    def record_clipboard(self, content: str) -> None:
        """Record clipboard change."""
        self.record_action("clipboard", {
            "length": len(content),
            "preview": content[:50] if content else "",
        })

    def set_current_task(self, task: str) -> None:
        """Set the current task being worked on."""
        self.current_task = task
        self.record_action("task_start", {"task": task})

    def get_recent_actions(self, limit: int = 10) -> list[FlowAction]:
        """Get recent actions."""
        return list(self.actions)[-limit:]

    def infer_intent(self) -> str | None:
        """Infer current intent from recent actions.

        Looks at recent actions to guess what the user is trying to do.
        """
        recent = self.get_recent_actions(5)
        if not recent:
            return None

        file_edits = [a for a in recent if a.action_type == "file_edit"]
        terminals = [a for a in recent if a.action_type == "terminal"]

        # Pattern: lots of file edits = refactoring
        if len(file_edits) >= 3:
            return "refactoring"

        # Pattern: file edit + tests = adding feature
        if file_edits and terminals:
            term = terminals[-1].details.get("command", "")
            if "test" in term:
                return "adding tests"

        # Pattern: just file edits = implementing
        if file_edits:
            return "implementing"

        # Pattern: just terminals = exploring/debugging
        if terminals:
            return "exploring"

        return None

    def get_context_summary(self) -> str:
        """Get a summary of current context."""
        recent = self.get_recent_actions(5)
        if not recent:
            return "No recent actions"

        lines = ["## Flow Context\n"]
        lines.append(f"Task: {self.current_task or 'Unknown'}\n")
        lines.append(f"Inferred intent: {self.infer_intent() or 'None'}\n")

        lines.append("\n### Recent Actions")
        for a in recent:
            at = a.action_type
            if at == "file_edit":
                fp = a.details.get("file_path", "?")
                lines.append(f"- Edited: {fp}")
            elif at == "terminal":
                cmd = a.details.get("command", "?")
                lines.append(f"- Terminal: {cmd}")
            elif at == "clipboard":
                lines.append(f"- Clipboard: {a.details.get('preview', '')[:30]}...")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear action history."""
        self.actions.clear()
        self.current_task = None


# Global instance
_flow_awareness: FlowAwareness | None = None


def get_flow_awareness() -> FlowAwareness:
    """Get or create the flow awareness instance."""
    global _flow_awareness
    if _flow_awareness is None:
        _flow_awareness = FlowAwareness()
    return _flow_awareness


def record_user_action(action_type: str, details: dict[str, Any] | None = None) -> None:
    """Record a user action."""
    get_flow_awareness().record_action(action_type, details)


def get_context_for_agent() -> str:
    """Get context summary for the agent."""
    return get_flow_awareness().get_context_summary()