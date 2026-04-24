"""Checkpoint tool for automatic undo capability."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CheckpointTool(Tool):
    """Create and manage checkpoints for undo capability.

    Uses git to snapshot the current state so changes can be undone.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "checkpoint"

    @property
    def description(self) -> str:
        return (
            "Create checkpoints to save current state, "
            "list saved checkpoints, and restore to a previous checkpoint. "
            "Useful for safe experimentation with undo capability."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "restore", "undo"],
                    "description": "Action to perform",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint ID to restore",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        action: str,
        checkpoint_id: str | None = None,
    ) -> ToolResult:
        """Execute checkpoint action."""
        from godspeed.agent.checkpoints import get_checkpoint_manager

        manager = get_checkpoint_manager(str(tool_context.cwd))

        if action == "create":
            checkpoint_id = manager.create_checkpoint()
            return ToolResult.ok(f"Created checkpoint: {checkpoint_id}")

        elif action == "list":
            checkpoints = manager.list_checkpoints()
            if not checkpoints:
                return ToolResult.ok("No checkpoints saved")

            lines = ["## Checkpoints"]
            for cp in checkpoints:
                lines.append(f"- {cp['id']}: {cp['description'] or cp['name']}")

            return ToolResult.ok("\n".join(lines))

        elif action == "restore":
            if not checkpoint_id:
                return ToolResult.failure("checkpoint_id required for restore")

            success = manager.restore_checkpoint(checkpoint_id)
            if success:
                return ToolResult.ok(f"Restored to checkpoint: {checkpoint_id}")
            return ToolResult.failure(f"Failed to restore checkpoint: {checkpoint_id}")

        elif action == "undo":
            success = manager.undo_last()
            if success:
                return ToolResult.ok("Undone last change")
            return ToolResult.failure("Failed to undo")

        return ToolResult.failure(f"Unknown action: {action}")
