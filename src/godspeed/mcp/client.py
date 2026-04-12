"""MCP client — connect to Model Context Protocol servers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPServerConfig:
    """Configuration for an MCP server connection.

    Supports two transport modes:

    **stdio** (default) — launches a local subprocess and communicates over
    stdin/stdout.  Requires ``command`` and optionally ``args``/``env``.

    **sse** — connects to a remote MCP server over HTTP (JSON-RPC).
    Requires ``url`` and optionally ``headers`` for auth tokens.

    Attributes:
        name: Human-readable server identifier used in tool prefixes.
        command: Executable path for stdio transport (ignored for sse).
        args: CLI arguments passed to *command* (stdio only).
        env: Extra environment variables for the subprocess (stdio only).
        transport: ``"stdio"`` (default) or ``"sse"``.
        url: Base URL of the remote MCP server (sse only, e.g.
            ``"http://localhost:3001"``).
        headers: HTTP headers sent with every request (sse only).  Useful
            for Bearer tokens: ``{"Authorization": "Bearer <tok>"}``.
    """

    def __init__(
        self,
        name: str,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: str = "stdio",
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.transport = transport
        self.url = url
        self.headers = headers


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

    Wraps the mcp library for stdio transport and MCPSSEClient for
    SSE/HTTP transport. Gracefully handles the case where dependencies
    are not installed.
    """

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._sse_clients: dict[str, Any] = {}
        self._stdio_available = self._check_stdio_available()
        self._sse_available = self._check_sse_available()

    @staticmethod
    def _check_stdio_available() -> bool:
        """Check whether the ``mcp`` package is installed (required for stdio)."""
        try:
            import mcp  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _check_sse_available() -> bool:
        """Check whether ``httpx`` is installed (required for SSE transport)."""
        try:
            import httpx  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def stdio_available(self) -> bool:
        """True when the ``mcp`` SDK is installed (needed for stdio transport)."""
        return self._stdio_available

    @property
    def available(self) -> bool:
        """True when at least one transport backend can be used.

        SSE transport only requires ``httpx`` (always bundled).  stdio
        transport requires the ``mcp`` package.
        """
        return self._stdio_available or self._sse_available

    async def connect(self, config: MCPServerConfig) -> list[MCPToolDefinition]:
        """Connect to an MCP server and discover its tools.

        Returns a list of tool definitions provided by the server.
        Supports both stdio and SSE transports based on config.transport.
        """
        if config.transport == "sse":
            return await self._connect_sse(config)
        return await self._connect_stdio(config)

    async def _connect_sse(self, config: MCPServerConfig) -> list[MCPToolDefinition]:
        """Connect to a remote MCP server via SSE/HTTP transport."""
        if not self._sse_available:
            logger.warning("httpx not installed — skipping SSE server %s", config.name)
            return []

        if config.url is None:
            logger.error("MCP SSE config missing url server=%s", config.name)
            return []

        try:
            from godspeed.mcp.sse_transport import MCPSSEClient

            client = MCPSSEClient(base_url=config.url, headers=config.headers)
            await client.connect()
            self._sse_clients[config.name] = client

            raw_tools = await client.list_tools()
            definitions: list[MCPToolDefinition] = []
            for tool in raw_tools:
                defn = MCPToolDefinition(
                    name=f"mcp_{config.name}_{tool.get('name', 'unknown')}",
                    description=tool.get("description", f"MCP tool: {tool.get('name')}"),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=config.name,
                )
                definitions.append(defn)

            logger.info(
                "MCP SSE connected server=%s tools=%d",
                config.name,
                len(definitions),
            )
            return definitions

        except Exception as exc:
            logger.error("MCP SSE connection failed server=%s error=%s", config.name, exc)
            return []

    async def _connect_stdio(self, config: MCPServerConfig) -> list[MCPToolDefinition]:
        """Connect to a local MCP server via stdio transport."""
        if not self._stdio_available:
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
        # Check SSE clients first
        sse_client = self._sse_clients.get(server_name)
        if sse_client is not None:
            return await sse_client.call_tool(tool_name, arguments)

        # Fall back to stdio session
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
        for sse_client in self._sse_clients.values():
            try:
                await sse_client.disconnect()
            except Exception as exc:
                logger.error("MCP SSE disconnect error: %s", exc)
        self._sse_clients.clear()
        self._connections.clear()
