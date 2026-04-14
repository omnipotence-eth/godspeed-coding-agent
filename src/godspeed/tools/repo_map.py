"""Repo map tool — structured codebase overview via tree-sitter."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)


class RepoMapTool(Tool):
    """Generate a symbol map of a directory using tree-sitter.

    Produces a compressed outline showing classes, functions, and methods
    with line numbers. Helps the agent understand codebase structure
    without reading every file.
    """

    @property
    def name(self) -> str:
        return "repo_map"

    @property
    def description(self) -> str:
        return (
            "Generate a symbol map of a directory showing classes, functions, "
            "and methods with line numbers. Uses tree-sitter for accurate parsing. "
            "Supports Python, JavaScript, TypeScript, and Go.\n\n"
            "Example: repo_map()\n"
            "Example: repo_map(path='src/', pattern='*.py')\n"
            "Example: repo_map(path='lib/', max_depth=3)"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to map (relative to project root). "
                        "Defaults to project root if empty."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py')",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth (default: 5)",
                },
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        from godspeed.context.repo_map import RepoMapper

        path_str = arguments.get("path", "")
        pattern = arguments.get("pattern", "")
        max_depth = arguments.get("max_depth", 5)

        if not isinstance(max_depth, int) or max_depth < 1:
            max_depth = 5

        if path_str:
            try:
                resolved = resolve_tool_path(path_str, context.cwd)
            except ValueError as exc:
                return ToolResult.failure(str(exc))
        else:
            resolved = context.cwd

        if not resolved.is_dir():
            return ToolResult.failure(f"Not a directory: {path_str or '.'}")

        mapper = RepoMapper()
        if not mapper.available:
            return ToolResult.failure(
                "tree-sitter not available. Install with: pip install godspeed[context]"
            )

        result = mapper.map_directory(resolved, max_depth=max_depth, pattern=pattern)
        return ToolResult.success(result)
