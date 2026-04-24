"""Command palette for quick access to commands and actions."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CommandPaletteTool(Tool):
    """Quick command palette for fast access.

    Similar to VS Code's Ctrl+Shift+P and Cursor's Cmd+K.
    Provides fuzzy search and quick execution of commands.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "command_palette"

    @property
    def description(self) -> str:
        return (
            "Open command palette for quick access to commands. "
            "Provides fuzzy search for commands and actions. "
            "Type to filter, Enter to execute."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Fuzzy search query",
                },
                "mode": {
                    "type": "string",
                    "enum": ["search", "execute", "list"],
                    "description": "Palette mode",
                },
            },
        }

    async def execute(
        self,
        tool_context: ToolContext,
        query: str = "",
        mode: str = "list",
    ) -> ToolResult:
        """Execute command palette action."""
        from difflib import SequenceMatcher

        commands = [
            {"id": "goto_definition", "label": "Go to Definition", "category": "Navigate"},
            {"id": "goto_reference", "label": "Find References", "category": "Navigate"},
            {"id": "find_files", "label": "Find Files", "category": "Search"},
            {"id": "grep_search", "label": "Search in Files", "category": "Search"},
            {"id": "format_document", "label": "Format Document", "category": "Edit"},
            {"id": "organize_imports", "label": "Organize Imports", "category": "Edit"},
            {"id": "rename_symbol", "label": "Rename Symbol", "category": "Refactor"},
            {"id": "extract_function", "label": "Extract to Function", "category": "Refactor"},
            {"id": "run_tests", "label": "Run Tests", "category": "Run"},
            {"id": "run_build", "label": "Run Build", "category": "Run"},
            {"id": "run_lint", "label": "Run Linter", "category": "Run"},
            {"id": "toggle_terminal", "label": "Toggle Terminal", "category": "View"},
            {"id": "toggle_sidebar", "label": "Toggle Sidebar", "category": "View"},
            {"id": "open_settings", "label": "Open Settings", "category": "Config"},
            {"id": "run_git_status", "label": "Git Status", "category": "Git"},
            {"id": "run_git_commit", "label": "Git Commit", "category": "Git"},
        ]

        if mode == "list":
            lines = ["## Command Palette\n"]
            lines.append("| Command | Category |")
            lines.append("|--------|----------|")
            for cmd in commands:
                lines.append(f"| {cmd['label']} | {cmd['category']} |")
            lines.append("\nUse mode=search with query to find commands.")
            return ToolResult.ok("\n".join(lines))

        if mode == "search" and query:
            query_lower = query.lower()

            def fuzzy_score(s: str) -> float:
                return SequenceMatcher(None, query_lower, s.lower()).ratio()

            scored = [
                (fuzzy_score(c["label"]), c) for c in commands
            ]
            scored = [c for s, c in scored if s > 0.3]
            scored.sort(key=lambda x: x["label"])

            if not scored:
                return ToolResult.ok(f"No commands match: {query}")

            lines = ["## Search Results\n"]
            for c in scored[:10]:
                lines.append(f"- {c['label']} ({c['category']})")

            return ToolResult.ok("\n".join(lines))

        if mode == "execute" and query:
            query_lower = query.lower()

            for c in commands:
                if query_lower in c["label"].lower():
                    return ToolResult.ok(
                        f"Executing: {c['label']} - Use the specific tool for this action."
                    )

            return ToolResult.failure(f"Command not found: {query}")

        return ToolResult.failure(f"Unknown mode: {mode}")