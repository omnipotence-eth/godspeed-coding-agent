"""MCP tool adapter — maps MCP tool definitions to Godspeed Tool ABC."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.mcp.client import MCPClient, MCPToolDefinition
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class MCPToolAdapter(Tool):
    """Adapts an MCP tool definition into a Godspeed Tool.

    All MCP tools default to HIGH risk since they execute external code
    from third-party servers.
    """

    def __init__(
        self,
        definition: MCPToolDefinition,
        mcp_client: MCPClient,
    ) -> None:
        self._definition = definition
        self._mcp_client = mcp_client
        # Strip mcp_{server}_ prefix to get the original tool name
        prefix = f"mcp_{definition.server_name}_"
        self._original_name = definition.name.removeprefix(prefix)

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        schema = self._definition.input_schema
        if not schema or not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        return schema

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the MCP tool by calling the remote server."""
        try:
            result = await self._mcp_client.call_tool(
                server_name=self._definition.server_name,
                tool_name=self._original_name,
                arguments=arguments,
            )
            if result.startswith("Error:"):
                return ToolResult.failure(result)
            return ToolResult.success(result)
        except Exception as exc:
            logger.error(
                "MCP tool execution failed tool=%s error=%s",
                self.name,
                exc,
                exc_info=True,
            )
            return ToolResult.failure(f"MCP tool '{self.name}' failed: {exc}")


def adapt_mcp_tools(
    definitions: list[MCPToolDefinition],
    mcp_client: MCPClient,
) -> list[MCPToolAdapter]:
    """Convert a list of MCP tool definitions into Godspeed tools."""
    return [MCPToolAdapter(defn, mcp_client) for defn in definitions]
