"""Tests for MCP client."""

from __future__ import annotations

from godspeed.mcp.client import MCPClient, MCPServerConfig, MCPToolDefinition


class TestMCPServerConfig:
    """Test MCP server configuration."""

    def test_defaults(self) -> None:
        config = MCPServerConfig(name="test", command="echo")
        assert config.name == "test"
        assert config.command == "echo"
        assert config.args == []
        assert config.env == {}
        assert config.transport == "stdio"

    def test_with_args(self) -> None:
        config = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "test"},
        )
        assert config.args == ["-y", "@modelcontextprotocol/server-github"]
        assert config.env == {"GITHUB_TOKEN": "test"}

    def test_sse_config(self) -> None:
        """SSE transport config stores url and headers."""
        config = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:3001",
            headers={"Authorization": "Bearer tok123"},
        )
        assert config.transport == "sse"
        assert config.url == "http://localhost:3001"
        assert config.headers == {"Authorization": "Bearer tok123"}
        # stdio fields get defaults
        assert config.command == ""
        assert config.args == []

    def test_transport_defaults_to_stdio(self) -> None:
        """Omitting transport field defaults to stdio (backward compat)."""
        config = MCPServerConfig(name="legacy", command="/usr/bin/server")
        assert config.transport == "stdio"
        assert config.url is None
        assert config.headers is None


class TestMCPToolDefinition:
    """Test tool definition data class."""

    def test_basic(self) -> None:
        defn = MCPToolDefinition(
            name="mcp_github_search",
            description="Search GitHub repos",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            server_name="github",
        )
        assert defn.name == "mcp_github_search"
        assert defn.server_name == "github"
        assert "query" in defn.input_schema["properties"]


class TestMCPClient:
    """Test MCP client behavior."""

    def test_availability_check(self) -> None:
        """Client reports availability based on installed transport backends."""
        client = MCPClient()
        # available is True when at least one backend (stdio or sse) is usable
        assert isinstance(client.available, bool)
        assert isinstance(client.stdio_available, bool)

    def test_disconnect_all(self) -> None:
        """Disconnect clears connections without error."""
        import asyncio

        # asyncio.run creates + tears down a fresh loop; avoids the
        # cross-test loop-state bleed caused by get_event_loop in
        # Python 3.12+.
        client = MCPClient()
        asyncio.run(client.disconnect_all())
