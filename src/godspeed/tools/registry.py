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
        self._description_overrides: dict[str, str] = {}  # tool_name -> override
        self._schema_cache: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if tool.name in self._tools:
            msg = f"Tool '{tool.name}' is already registered"
            raise ValueError(msg)
        self._tools[tool.name] = tool
        self._schema_cache = None  # Invalidate cache
        logger.debug("Registered tool: %s (risk=%s)", tool.name, tool.risk_level)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def update_description(self, tool_name: str, description: str) -> bool:
        """Set a runtime description override for a tool.

        The override is used in get_schemas() instead of the tool's built-in
        description. Used by the self-evolution system to hot-swap descriptions.

        Returns:
            True if the tool exists and the override was set.
        """
        if tool_name not in self._tools:
            return False
        self._description_overrides[tool_name] = description
        self._schema_cache = None  # Invalidate cache
        logger.debug("Description override set tool=%s len=%d", tool_name, len(description))
        return True

    def clear_description_override(self, tool_name: str) -> None:
        """Remove a description override, reverting to the built-in description."""
        if self._description_overrides.pop(tool_name, None) is not None:
            self._schema_cache = None  # Invalidate cache

    def get_description(self, tool_name: str) -> str | None:
        """Get the effective description for a tool (override or built-in)."""
        if tool_name in self._description_overrides:
            return self._description_overrides[tool_name]
        tool = self._tools.get(tool_name)
        return tool.description if tool else None

    def get_schemas(self) -> list[dict[str, Any]]:
        """Generate tool schemas in the format expected by LLM APIs.

        Returns a list of tool definitions compatible with OpenAI/Anthropic
        function calling format (LiteLLM normalizes this). Uses description
        overrides from the self-evolution system when available.

        Results are cached until a tool is registered or a description
        override changes.
        """
        if self._schema_cache is not None:
            return self._schema_cache

        schemas = []
        for tool in self._tools.values():
            description = self._description_overrides.get(tool.name, tool.description)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": tool.get_schema(),
                    },
                }
            )
        self._schema_cache = schemas
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
