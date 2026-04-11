"""Tool registry — discovery, schema generation, and dispatch."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import Tool, ToolCall, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available tools.

    Handles tool registration, schema generation for LLM APIs,
    and dispatching tool calls to the correct implementation.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if tool.name in self._tools:
            msg = f"Tool '{tool.name}' is already registered"
            raise ValueError(msg)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (risk=%s)", tool.name, tool.risk_level)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Generate tool schemas in the format expected by LLM APIs.

        Returns a list of tool definitions compatible with OpenAI/Anthropic
        function calling format (LiteLLM normalizes this).
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.get_schema(),
                    },
                }
            )
        return schemas

    async def dispatch(self, tool_call: ToolCall, context: ToolContext) -> ToolResult:
        """Dispatch a tool call to the correct tool implementation.

        Args:
            tool_call: The tool call to execute.
            context: Execution context.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            return ToolResult.failure(
                f"Unknown tool: '{tool_call.tool_name}'. "
                f"Available: {', '.join(sorted(self._tools.keys()))}"
            )

        try:
            return await tool.execute(tool_call.arguments, context)
        except Exception as exc:
            logger.error(
                "Tool execution failed tool=%s error=%s",
                tool_call.tool_name,
                exc,
                exc_info=True,
            )
            return ToolResult.failure(f"Tool '{tool_call.tool_name}' failed: {exc}")
