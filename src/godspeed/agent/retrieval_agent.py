"""Retrieval sub-agent — focused code exploration with structured file:line-range output."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

RETRIEVAL_SYSTEM_PROMPT = """\
You are a retrieval specialist. Answer questions about the codebase by searching \
and reading files. You have access to read-only tools only.

## Output Format
Return your findings as a structured list:

file:path/to/file.py:10-25   -- function FooBar (brief description)
file:path/to/file.py:42-42   -- relevant line

For each finding, include the file path, line range, and a brief description.

## Tools
- file_read -- read file contents
- grep_search -- search for patterns in code
- glob_search -- find files by name pattern
- repo_map -- get repository structure
- code_search -- semantic code search

## Guidelines
- Be thorough: search multiple patterns if the first pass doesn't find enough.
- Be concise: return only the most relevant results.
- Do NOT modify any files or run shell commands.
- Do NOT spawn further sub-agents.
- Return "No results found." if nothing matches.
"""

RETRIEVAL_READ_ONLY_TOOLS = {
    "file_read",
    "grep_search",
    "glob_search",
    "repo_map",
    "code_search",
    "web_search",
    "web_fetch",
}


class RetrievalSubAgentTool(Tool):
    """Spawn a retrieval sub-agent for focused code exploration.

    The retrieval sub-agent uses only read-only tools and returns
    structured ``file:line-range`` results. Useful when the main
    agent needs to delegate a deep code search without polluting
    its own conversation context.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator

    @property
    def name(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return (
            "Launch a focused retrieval search of the codebase. "
            "The retrieval agent uses read-only tools and returns "
            "structured file:line-range results. "
            "Use for deep code searches, understanding APIs, or "
            "finding all usages of a function."
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
                    "description": "The search query or code exploration task",
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments.get("query", "")
        if not query:
            return ToolResult.failure("query is required for retrieval")

        result = await self._coordinator.spawn_retrieval(query)
        return ToolResult.success(result)
