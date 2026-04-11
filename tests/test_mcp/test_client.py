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
        """Client reports availability based on mcp package."""
        client = MCPClient()
        # Should be True or False depending on whether mcp is installed
        assert isinstance(client.available, bool)

    def test_disconnect_all(self) -> None:
        """Disconnect clears connections without error."""
        import asyncio

        client = MCPClient()
        asyncio.get_event_loop().run_until_complete(client.disconnect_all())
