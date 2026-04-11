"""MCP client — connect to Model Context Protocol servers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPServerConfig:
    """Configuration for an MCP server connection."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: str = "stdio",
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.transport = transport


class MCPToolDefinition:
    """A tool definition discovered from an MCP server."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        server_name: str,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.server_name = server_name


class MCPClient:
    """Client for discovering and calling tools on MCP servers.

    Wraps the mcp library for stdio transport. Gracefully handles
    the case where the mcp package is not installed.
    """

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import mcp  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    async def connect(self, config: MCPServerConfig) -> list[MCPToolDefinition]:
        """Connect to an MCP server and discover its tools.

        Returns a list of tool definitions provided by the server.
        """
        if not self._available:
            logger.warning("MCP package not installed — skipping server %s", config.name)
            return []

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
            )

            async with (
                stdio_client(params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()

                # Store session reference for later calls
                self._connections[config.name] = session

                # Discover tools
                tools_response = await session.list_tools()
                definitions = []
                for tool in tools_response.tools:
                    defn = MCPToolDefinition(
                        name=f"mcp_{config.name}_{tool.name}",
                        description=tool.description or f"MCP tool: {tool.name}",
                        input_schema=(tool.inputSchema if hasattr(tool, "inputSchema") else {}),
                        server_name=config.name,
                    )
                    definitions.append(defn)

                logger.info(
                    "MCP connected server=%s tools=%d",
                    config.name,
                    len(definitions),
                )
                return definitions

        except Exception as exc:
            logger.error("MCP connection failed server=%s error=%s", config.name, exc)
            return []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool on a connected MCP server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Original tool name (without mcp_ prefix).
            arguments: Tool arguments.

        Returns:
            The tool result as a string.
        """
        session = self._connections.get(server_name)
        if session is None:
            return f"Error: MCP server '{server_name}' is not connected"

        try:
            result = await session.call_tool(tool_name, arguments)
            # Extract text content from result
            if hasattr(result, "content"):
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                return "\n".join(parts) if parts else str(result)
            return str(result)
        except Exception as exc:
            logger.error(
                "MCP tool call failed server=%s tool=%s error=%s",
                server_name,
                tool_name,
                exc,
            )
            return f"Error: MCP tool call failed — {exc}"

    async def disconnect_all(self) -> None:
        """Close all MCP server connections."""
        self._connections.clear()
