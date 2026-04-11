"""File edit tool — search/replace with fuzzy matching fallback."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

# Minimum similarity ratio for fuzzy matching
FUZZY_THRESHOLD = 0.8


class FileEditTool(Tool):
    """Edit a file using search/replace.

    The most reliable edit format across all LLM models. Finds the exact
    old_string in the file and replaces it with new_string. Falls back to
    fuzzy matching (difflib.SequenceMatcher) when exact match fails due
    to whitespace drift.
    """

    @property
    def name(self) -> str:
        return "file_edit"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_string with new_string. "
            "The old_string must be unique in the file (include surrounding context "
            "to ensure uniqueness). Read the file first to get exact content."
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
                    "description": "Path to the file to edit (relative to project root)",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace (must be unique in file)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        replace_all = arguments.get("replace_all", False)

        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")
        if not isinstance(old_string, str) or not old_string:
            return ToolResult.failure("old_string must be a non-empty string")
        if not isinstance(new_string, str):
            return ToolResult.failure(
                f"new_string must be a string, got {type(new_string).__name__}"
            )
        if old_string == new_string:
            return ToolResult.failure("old_string and new_string must be different")

        try:
            resolved = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not resolved.exists():
            return ToolResult.failure(f"File not found: {file_path_str}")

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(f"Cannot edit binary file: {file_path_str}")

        # Normalize line endings to LF for consistent matching
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Try exact match first
        if old_string in content:
            count = content.count(old_string)
            if count > 1 and not replace_all:
                return ToolResult.failure(
                    f"old_string matches {count} locations in {file_path_str}. "
                    "Include more surrounding context to make it unique, "
                    "or set replace_all=true."
                )
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1
        else:
            # Fuzzy matching fallback
            match_result = _fuzzy_find(content, old_string)
            if match_result is None:
                return ToolResult.failure(
                    f"old_string not found in {file_path_str}. "
                    "Read the file first to get the exact content."
                )
            start, end, ratio = match_result
            new_content = content[:start] + new_string + content[end:]
            replacements = 1
            logger.info(
                "Fuzzy match used ratio=%.2f path=%s",
                ratio,
                file_path_str,
            )

        resolved.write_text(new_content, encoding="utf-8")
        logger.info(
            "Edited file path=%s replacements=%d",
            file_path_str,
            replacements,
        )
        return ToolResult.success(f"Replaced {replacements} occurrence(s) in {file_path_str}")


def _fuzzy_find(content: str, search: str) -> tuple[int, int, float] | None:
    """Find the best fuzzy match for search in content.

    Uses a sliding window approach with SequenceMatcher.
    Returns (start, end, ratio) of the best match, or None if below threshold.
    """
    # Normalize line endings — caller should also normalize, but be defensive
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    search = search.replace("\r\n", "\n").replace("\r", "\n")

    search_lines = search.splitlines()
    content_lines = content.splitlines()

    if not search_lines or not content_lines:
        return None

    window_size = len(search_lines)
    best_ratio = 0.0
    best_start_line = 0

    for i in range(len(content_lines) - window_size + 1):
        window = content_lines[i : i + window_size]
        window_text = "\n".join(window)
        ratio = SequenceMatcher(None, search, window_text).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start_line = i

    if best_ratio < FUZZY_THRESHOLD:
        return None

    # Convert line indices to character positions using join (handles edge cases)
    lines_before = content_lines[:best_start_line]
    start_pos = len("\n".join(lines_before)) + (1 if lines_before else 0)
    matched_lines = content_lines[best_start_line : best_start_line + window_size]
    end_pos = start_pos + len("\n".join(matched_lines))

    # Clamp to content length
    if end_pos > len(content):
        end_pos = len(content)

    return start_pos, end_pos, best_ratio
