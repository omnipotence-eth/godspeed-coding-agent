"""Grep search tool — search file contents by regex pattern."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.excludes import is_excluded

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LINES = 2
MAX_MATCHES = 200


def _search_file(
    file_path: Path,
    compiled: re.Pattern[str],
    context_lines: int,
) -> list[str]:
    """Search a single file for regex matches. Returns formatted match blocks."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return []

    lines = content.splitlines()
    match_indices: set[int] = set()

    for i, line in enumerate(lines):
        if compiled.search(line):
            match_indices.add(i)

    if not match_indices:
        return []

    # Build context ranges
    display_indices: set[int] = set()
    for idx in match_indices:
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        for j in range(start, end):
            display_indices.add(j)

    # Group into contiguous blocks
    sorted_indices = sorted(display_indices)
    blocks: list[list[str]] = []
    current_block: list[str] = []
    prev = -2

    for idx in sorted_indices:
        if idx != prev + 1 and current_block:
            blocks.append(current_block)
            current_block = []
        marker = ">" if idx in match_indices else " "
        current_block.append(f"{marker} {idx + 1:>5}\t{lines[idx]}")
        prev = idx

    if current_block:
        blocks.append(current_block)

    return ["\n".join(block) for block in blocks]


class GrepSearchTool(Tool):
    """Search file contents by regex pattern.

    Uses Python's re module to search files. Shows matching lines with
    line numbers and surrounding context. Respects common exclude patterns.
    """

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return (
            "Search file contents using a regex pattern. "
            "Shows matching lines with line numbers and surrounding context. "
            "Excludes common directories like node_modules, .venv, __pycache__, .git."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search in (relative to project root). "
                        "Default: project root."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Glob pattern to filter which files to search "
                        "(e.g., '*.py', '*.{ts,tsx}'). Default: all files."
                    ),
                },
                "context_lines": {
                    "type": "integer",
                    "description": (
                        f"Lines of context around each match. Default: {DEFAULT_CONTEXT_LINES}"
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return ToolResult.failure("pattern is required")

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return ToolResult.failure(f"Invalid regex pattern: {exc}")

        raw_context = arguments.get("context_lines", DEFAULT_CONTEXT_LINES)
        if not isinstance(raw_context, int):
            try:
                raw_context = int(raw_context)
            except (TypeError, ValueError):
                return ToolResult.failure("context_lines must be an integer")
        context_lines = max(0, raw_context)
        file_glob = arguments.get("glob", "")

        # Resolve search path
        search_path = context.cwd
        path_arg = arguments.get("path")
        if path_arg:
            candidate = Path(path_arg)
            search_path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (context.cwd / candidate).resolve()
            )

        try:
            search_path.resolve().relative_to(context.cwd.resolve())
        except ValueError:
            return ToolResult.failure(f"Search path must be within project directory: {path_arg}")

        if not search_path.exists():
            return ToolResult.failure(f"Search path does not exist: {search_path}")

        logger.info("grep_search pattern=%r path=%s glob=%r", pattern, search_path, file_glob)

        # Collect files to search
        if search_path.is_file():
            files = [search_path]
        else:
            glob_pattern = file_glob if file_glob else "**/*"
            files = [
                p
                for p in search_path.glob(glob_pattern)
                if p.is_file() and not is_excluded(p.relative_to(search_path))
            ]

        # Search files and collect results
        total_matches = 0
        output_sections: list[str] = []

        for file_path in sorted(files):
            blocks = _search_file(file_path, compiled, context_lines)
            if blocks:
                try:
                    rel_path = file_path.relative_to(context.cwd)
                except ValueError:
                    rel_path = file_path
                header = f"--- {rel_path} ---"
                section = header + "\n" + "\n---\n".join(blocks)
                output_sections.append(section)
                total_matches += len(blocks)

                if total_matches >= MAX_MATCHES:
                    break

        if not output_sections:
            return ToolResult.success(f"No matches found for pattern '{pattern}'")

        output = "\n\n".join(output_sections)
        if total_matches >= MAX_MATCHES:
            output += f"\n\n... (truncated at {MAX_MATCHES} match groups)"

        return ToolResult.success(output)
