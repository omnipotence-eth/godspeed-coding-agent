"""Glob search tool — find files by glob pattern."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.excludes import is_excluded

logger = logging.getLogger(__name__)

MAX_RESULTS = 500


class GlobSearchTool(Tool):
    """Find files by glob pattern.

    Uses pathlib.Path.glob() to match files. Returns paths sorted by
    modification time (most recent first). Respects common exclude
    patterns (node_modules, .venv, __pycache__, .git, etc.).
    """

    @property
    def name(self) -> str:
        return "glob_search"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern (e.g., '**/*.py'). "
            "Returns file paths sorted by modification time (newest first). "
            "Excludes common directories like node_modules, .venv, __pycache__, .git.\n\n"
            "Example: glob_search(pattern='**/*.py')\n"
            "Example: glob_search(pattern='src/**/*.ts', path='frontend/')\n"
            "Example: glob_search(pattern='**/test_*.py')"
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
                    "description": "Glob pattern to match files (e.g., '**/*.py', 'src/**/*.ts')",
                    "examples": ["**/*.py", "src/**/*.ts", "**/test_*.py"],
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search in (relative to project root). Default: project root."
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return ToolResult.failure("pattern is required")

        search_root = context.cwd
        path_arg = arguments.get("path")
        if path_arg:
            candidate = Path(path_arg).expanduser()
            search_root = (
                candidate.resolve()
                if candidate.is_absolute()
                else (context.cwd / candidate).resolve()
            )

        try:
            search_root.resolve().relative_to(context.cwd.resolve())
        except ValueError:
            return ToolResult.failure(f"Search path must be within project directory: {path_arg}")

        if not search_root.exists():
            return ToolResult.failure(f"Search path does not exist: {search_root}")
        if not search_root.is_dir():
            return ToolResult.failure(f"Search path is not a directory: {search_root}")

        logger.info("glob_search pattern=%r root=%s", pattern, search_root)

        def _safe_filter(p: Path) -> bool:
            """Filter a path, skipping inaccessible files."""
            try:
                if not p.is_file():
                    return False
                return not is_excluded(p.relative_to(search_root))
            except (OSError, PermissionError, ValueError):
                return False

        try:
            matches = [p for p in search_root.glob(pattern) if _safe_filter(p)]
        except ValueError as exc:
            return ToolResult.failure(f"Invalid glob pattern: {exc}")

        # Sort by modification time, newest first (skip inaccessible files)
        with contextlib.suppress(OSError):
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not matches:
            return ToolResult.success(f"No files found matching '{pattern}'")

        truncated = len(matches) > MAX_RESULTS
        matches = matches[:MAX_RESULTS]

        # Format as relative paths from the search root
        lines = [str(p.relative_to(search_root)) for p in matches]
        output = "\n".join(lines)

        if truncated:
            output += f"\n\n... (truncated to {MAX_RESULTS} results)"

        return ToolResult.success(output)
