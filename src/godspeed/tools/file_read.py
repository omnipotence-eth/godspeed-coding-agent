"""File read tool — read file contents with line numbers."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_KB = 5000
MAX_LINES = 2000


class FileReadTool(Tool):
    """Read file contents with line numbers.

    Returns file contents prefixed with line numbers for precise editing.
    Supports offset and limit for reading portions of large files.
    """

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read a file's contents with line numbers. "
            "Supports offset and limit for reading portions of large files. "
            "Always read a file before editing it."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to project root)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based). Default: 1",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of lines to read. Default: {MAX_LINES}",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")

        try:
            file_path = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        raw_offset = arguments.get("offset", 1)
        raw_limit = arguments.get("limit", MAX_LINES)
        if not isinstance(raw_offset, int):
            try:
                raw_offset = int(raw_offset)
            except (TypeError, ValueError):
                return ToolResult.failure(
                    f"offset must be an integer, got {type(raw_offset).__name__}"
                )
        if not isinstance(raw_limit, int):
            try:
                raw_limit = int(raw_limit)
            except (TypeError, ValueError):
                return ToolResult.failure(
                    f"limit must be an integer, got {type(raw_limit).__name__}"
                )
        offset = max(1, raw_offset)
        limit = min(raw_limit, MAX_LINES)

        if not file_path.exists():
            return ToolResult.failure(f"File not found: {file_path}")

        if not file_path.is_file():
            return ToolResult.failure(f"Not a file: {file_path}")

        # Size check
        size_kb = file_path.stat().st_size / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            return ToolResult.failure(
                f"File too large: {size_kb:.0f}KB (max {MAX_FILE_SIZE_KB}KB). "
                "Use offset and limit to read portions."
            )

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(f"Cannot read binary file: {file_path}")

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset and limit
        start_idx = offset - 1
        end_idx = start_idx + limit
        selected = lines[start_idx:end_idx]

        if not selected:
            return ToolResult.failure(
                f"No content at offset {offset}. File has {total_lines} lines."
            )

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=offset):
            numbered.append(f"{i:>6}\t{line}")

        output = "\n".join(numbered)

        if end_idx < total_lines:
            output += f"\n\n... ({total_lines - end_idx} more lines)"

        return ToolResult.success(output)
