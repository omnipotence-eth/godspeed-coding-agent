"""Semantic code search tool backed by the codebase index."""

from __future__ import annotations

from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult


class CodeSearchTool(Tool):
    """Semantic code search using the codebase vector index.

    Falls back gracefully if the index is unavailable or building.
    """

    def __init__(self, index: Any) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return (
            "Search the codebase using natural language queries. "
            "Returns relevant code snippets ranked by semantic similarity."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute semantic code search."""
        query = arguments.get("query", "")
        if not query:
            return ToolResult.failure("query is required")

        top_k = arguments.get("top_k", 5)

        if not self._index.is_available:
            return ToolResult.failure(
                "Codebase index not available. Install with: pip install godspeed[index]"
            )

        if self._index.is_building:
            return ToolResult.ok("Index is being built. Use grep_search for now.")

        results = self._index.search(query, top_k=top_k)
        if not results:
            return ToolResult.ok("No results found.")

        lines = []
        for r in results:
            lines.append(f"## {r.file_path} (lines {r.start_line}-{r.end_line}, score: {r.score})")
            # Truncate content to 10 lines
            content_lines = r.content.splitlines()
            if len(content_lines) > 10:
                lines.extend(content_lines[:10])
                lines.append(f"  ... ({len(content_lines) - 10} more lines)")
            else:
                lines.extend(content_lines)
            lines.append("")

        return ToolResult.ok("\n".join(lines))
