"""File move tool — rename or relocate files."""

from __future__ import annotations

import logging
import os
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)


class FileMoveTool(Tool):
    """Move or rename a file or directory.

    Uses os.replace() for atomic rename (works cross-filesystem on Windows).
    Use force=true to overwrite an existing destination.
    """

    produces_diff = True

    @property
    def name(self) -> str:
        return "file_move"

    @property
    def description(self) -> str:
        return (
            "Move or rename a file/directory. "
            "Use force=true to overwrite an existing destination. "
            "Both source and destination must be within the project directory.\n\n"
            "Example: file_move(source='src/old_name.py', destination='src/new_name.py')\n"
            "Example: file_move(source='draft.md', destination='archive/draft.md', force=true)"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the file/directory to move (relative to project root)",
                    "examples": ["src/old_name.py", "draft.md"],
                },
                "destination": {
                    "type": "string",
                    "description": "New path for the file/directory (relative to project root)",
                    "examples": ["src/new_name.py", "archive/draft.md"],
                },
                "force": {
                    "type": "boolean",
                    "description": "Overwrite destination if it already exists (default: false)",
                },
            },
            "required": ["source", "destination"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        source_str = arguments.get("source", "")
        destination_str = arguments.get("destination", "")
        force = arguments.get("force", False)

        if not isinstance(source_str, str) or not source_str:
            return ToolResult.failure("source must be a non-empty string")
        if not isinstance(destination_str, str) or not destination_str:
            return ToolResult.failure("destination must be a non-empty string")
        if not isinstance(force, bool):
            return ToolResult.failure(f"force must be a boolean, got {type(force).__name__}")

        source_str = source_str.strip()
        destination_str = destination_str.strip()

        if source_str == destination_str:
            return ToolResult.failure(f"source and destination are the same path: '{source_str}'")

        # Resolve both paths within project directory
        try:
            source_resolved = resolve_tool_path(source_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        try:
            dest_resolved = resolve_tool_path(destination_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        # Validate source exists
        if not source_resolved.exists():
            return ToolResult.failure(
                f"Source not found: '{source_str}' (resolved to '{source_resolved}')"
            )

        # Validate destination
        if dest_resolved.exists():
            if not force:
                return ToolResult.failure(
                    f"Destination already exists: '{destination_str}' (use force=true to overwrite)"
                )
            elif not dest_resolved.is_file() and source_resolved.is_file():
                return ToolResult.failure(
                    f"Cannot overwrite directory '{destination_str}' with file '{source_str}'"
                )
            elif not dest_resolved.is_dir() and source_resolved.is_dir():
                return ToolResult.failure(
                    f"Cannot overwrite file '{destination_str}' with directory '{source_str}'"
                )

        # Diff review gate — let the human approve/reject before moving
        if context.diff_reviewer is not None:
            decision = await context.diff_reviewer.review(
                tool_name=self.name,
                path=destination_str,
                before=f"{source_str} (move to {destination_str})",
                after=f"{destination_str}",
            )
            if decision != "accept":
                logger.info(
                    "Move rejected by diff_reviewer: source=%s dest=%s decision=%s",
                    source_str,
                    destination_str,
                    decision,
                )
                return ToolResult.failure(
                    f"Move rejected by reviewer for {source_str} -> {destination_str} "
                    "— no changes made."
                )

        # Ensure parent directory exists
        dest_resolved.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.replace(source_resolved, dest_resolved)
            logger.info(
                "Moved file source=%s destination=%s",
                source_str,
                destination_str,
            )
            return ToolResult.success(f"Moved '{source_str}' to '{destination_str}'")
        except OSError as exc:
            return ToolResult.failure(
                f"Failed to move '{source_str}' to '{destination_str}': {exc}"
            )
