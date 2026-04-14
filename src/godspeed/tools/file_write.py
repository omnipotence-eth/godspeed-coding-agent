"""File write tool — create new files."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)


class FileWriteTool(Tool):
    """Create or overwrite a file with new content.

    Use file_edit for modifying existing files.
    Use file_write only for creating new files or complete rewrites.
    """

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file and parent directories if needed. "
            "Overwrites existing content. Use file_edit for precise modifications.\n\n"
            "Example: file_write(file_path='src/utils.py', content='def helper():\\n    pass')\n"
            "Example: file_write(file_path='config.json', content='{\"debug\": true}')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write (relative to project root)",
                    "examples": ["src/utils.py", "config.json", "tests/test_new.py"],
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")
        if not isinstance(content, str):
            return ToolResult.failure(f"content must be a string, got {type(content).__name__}")

        try:
            resolved = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        try:
            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            resolved.write_text(content, encoding="utf-8")

            logger.info("Wrote file path=%s size=%d", resolved, len(content))
            return ToolResult.success(f"Wrote {len(content)} bytes to {file_path_str}")

        except OSError as exc:
            return ToolResult.failure(f"Failed to write {file_path_str}: {exc}")
