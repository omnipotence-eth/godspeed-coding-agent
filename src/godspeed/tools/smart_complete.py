"""Smart code completion tool — intelligent code suggestions."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SmartCompleteTool(Tool):
    """Generate intelligent code completions.

    Uses the LLM to suggest code based on the surrounding context,
    file patterns, and similar files in the codebase.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "smart_complete"

    @property
    def description(self) -> str:
        return (
            "Generate intelligent code completions based on context. "
            "Analyzes the file, finds similar patterns, and suggests completions. "
            "Useful for autocomplete, boilerplate, and pattern-based code generation."
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
                    "description": "File to get completion for",
                },
                "cursor_line": {
                    "type": "integer",
                    "description": "Line number for cursor position",
                },
                "prefix": {
                    "type": "string",
                    "description": "Code prefix before cursor",
                },
                "suffix": {
                    "type": "string",
                    "description": "Code suffix after cursor",
                },
                "num_suggestions": {
                    "type": "integer",
                    "description": "Number of suggestions to generate",
                },
            },
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        cursor_line: int = 1,
        prefix: str = "",
        suffix: str = "",
        num_suggestions: int = 3,
    ) -> ToolResult:
        """Generate smart completions."""
        from godspeed.llm.client import get_default_client

        if not file_path:
            return ToolResult.failure("file_path required")

        import os

        if not os.path.exists(file_path):
            return ToolResult.failure(f"File not found: {file_path}")

        try:
            with open(file_path) as f:
                content = f.read()
        except Exception as exc:
            return ToolResult.failure(f"Cannot read file: {exc}")

        lines = content.split("\n")

        context_before = "\n".join(lines[max(0, cursor_line - 20) : cursor_line])
        context_after = "\n".join(lines[cursor_line : cursor_line + 10])

        prompt = f"""Complete the following code. Suggest {num_suggestions} options.

File: {file_path}

Context before:
```
{context_before}
```

Prefix (what user typed):
```
{prefix}
```

Context after:
```
{context_after}
```

Provide {num_suggestions} completion options, each 1-5 lines max.
Format as:
```
1. <completion>
2. <completion>
...
```"""

        try:
            client = get_default_client()
            if client is None:
                return ToolResult.failure("No LLM client available")

            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=tool_context.model or "gpt-4",
            )

            suggestion_text = response.content if hasattr(response, "content") else str(response)

            return ToolResult.ok(
                f"## Completions for {file_path}:{cursor_line}\n\n{suggestion_text}"
            )

        except Exception as exc:
            logger.warning("smart_complete failed: %s", exc)
            return ToolResult.failure(f"Completion failed: {exc}")
