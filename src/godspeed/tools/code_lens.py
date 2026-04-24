"""Intelligent code actions - inline refactoring suggestions."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CodeLensTool(Tool):
    """Provide intelligent code actions inline.

    Similar to VS Code's CodeLens and quick actions.
    Suggests refactoring and code improvements based
    on the code context.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "code_lens"

    @property
    def description(self) -> str:
        return (
            "Get intelligent code actions for the current context. "
            "Suggests refactoring, fixes, and improvements based on "
            "code analysis."
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
                    "description": "File to analyze",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number for context",
                },
                "action_type": {
                    "type": "string",
                    "enum": ["suggest", "apply", "list"],
                    "description": "Action to perform",
                },
            },
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        line: int = 1,
        action_type: str = "suggest",
    ) -> ToolResult:
        """Execute code lens action."""
        import os

        if not file_path or not os.path.exists(file_path):
            return ToolResult.failure(f"File not found: {file_path}")

        try:
            with open(file_path) as f:
                content = f.read()
        except Exception as exc:
            return ToolResult.failure(f"Cannot read file: {exc}")

        actions = self._analyze_code(content, file_path)

        if action_type == "list":
            lines = ["## Available Code Actions\n"]
            for a in actions:
                lines.append(f"- {a['title']}: {a['description']}")
            return ToolResult.ok("\n".join(lines))

        if action_type == "suggest" or action_type == "apply":
            if not actions:
                return ToolResult.ok("No code actions available for this file.")

            lines = ["## Code Actions\n"]
            for i, a in enumerate(actions, 1):
                lines.append(f"{i}. **{a['title']}**")
                lines.append(f"   {a['description']}")
                lines.append("")

            return ToolResult.ok("\n".join(lines))

        return ToolResult.failure(f"Unknown action: {action_type}")

    def _analyze_code(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Analyze code and suggest actions."""
        import ast

        actions = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return actions

        file_ext = file_path.split(".")[-1] if "." in file_path else ""

        # Analyze Python files
        if file_ext in ("py", "pyi"):
            # Check for long functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, 'body') and len(node.body) > 50:
                        actions.append({
                            "title": f"Split function '{node.name}'",
                            "description": f"Function has {len(node.body)} lines - consider splitting",
                            "action": "refactor",
                        })

            # Check for complex conditionals
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if isinstance(node.test, ast.BoolOp):
                        actions.append({
                            "title": "Simplify conditional",
                            "description": "Complex boolean expression can be simplified",
                            "action": "simplify",
                        })

            # Check for unused imports
            imported_names = set()
            used_names = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)

            unused = imported_names - used_names
            if unused:
                actions.append({
                    "title": "Remove unused imports",
                    "description": f"Found unused: {', '.join(list(unused)[:3])}",
                    "action": "remove_unused",
                })

        # Check for TODOs
        if "TODO" in content or "FIXME" in content:
            actions.append({
                "title": "Address TODOs",
                "description": "File contains TODO/FIXME comments",
                "action": "show_todos",
            })

        # Check for long lines
        for i, line in enumerate(content.split("\n"), 1):
            if len(line) > 120:
                actions.append({
                    "title": "Format long lines",
                    "description": f"Line {i} exceeds 120 characters",
                    "action": "format",
                })
                break  # Just report once

        return actions


class InlineLensTool(Tool):
    """Get inline actions directly in the editor context.

    Similar to VS Code's inline actions that appear
    on hover or at specific code locations.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "inline_lens"

    @property
    def description(self) -> str:
        return (
            "Get inline actions for code at a specific location. "
            "Appears directly in the code like VS Code lens."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "line": {"type": "integer"},
                "column": {"type": "integer"},
            },
            "required": ["file_path", "line"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        line: int = 1,
        column: int = 0,
    ) -> ToolResult:
        """Get inline actions."""
        import os

        if not os.path.exists(file_path):
            return ToolResult.failure(f"File not found: {file_path}")

        try:
            with open(file_path) as f:
                lines = f.readlines()
        except Exception as exc:
            return ToolResult.failure(f"Cannot read: {exc}")

        if line < 1 or line > len(lines):
            return ToolResult.failure(f"Line {line} out of range")

        current_line = lines[line - 1].rstrip()

        # Analyze line context
        actions = []

        # Extract symbol at position
        import re

        symbol_match = re.search(r'[\w]+', current_line[column:]) if column < len(current_line) else None
        if symbol_match:
            symbol = symbol_match.group()
            actions.append({
                "title": f"Rename '{symbol}'",
                "command": "rename_symbol",
            })

        actions.append({"title": "Go to definition", "command": "goto_definition"})
        actions.append({"title": "Find references", "command": "find_references"})
        actions.append({"title": "Peek definition", "command": "peek_definition"})

        lines = ["## Inline Actions\n"]
        for a in actions:
            lines.append(f"- {a['title']}")

        lines.append(f"\n**Current**: `{current_line[:50]}`")

        return ToolResult.ok("\n".join(lines))
