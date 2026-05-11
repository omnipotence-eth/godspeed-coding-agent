"""Tests for MCP SSE/HTTP transport."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from godspeed.mcp.client import MCPClient, MCPServerConfig
from godspeed.mcp.sse_transport import MCPSSEClient


def _make_response(
    json_data: dict | list | None = None,
    status_code: int = 200,
    content_length: int | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    import json as _json

    body = _json.dumps(json_data or {}).encode()
    headers = {"content-type": "application/json"}
    if content_length is not None:
        headers["content-length"] = str(content_length)
    resp = httpx.Response(
        status_code=status_code,
        content=body,
        headers=headers,
        request=httpx.Request("POST", "http://test"),
    )
    return resp


class TestMCPSSETransport:
    """Tests for MCPSSEClient."""

    @pytest.mark.asyncio
    async def test_sse_client_connect(self) -> None:
        """Mock httpx, verify POST to /initialize is made."""
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_make_response({}))

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()

        mock_http.post.assert_called_once_with("/initialize", json={})

    @pytest.mark.asyncio
    async def test_sse_client_list_tools(self) -> None:
        """Mock response with tool definitions, verify parsing."""
        tools_payload = {
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
                {
                    "name": "fetch",
                    "description": "Fetch a URL",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                    },
                },
            ]
        }

        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),  # /initialize
                _make_response(tools_payload),  # /tools/list
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "search"
        assert tools[1]["name"] == "fetch"
        assert "query" in tools[0]["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_sse_client_call_tool(self) -> None:
        """Mock POST, verify result extraction from content array."""
        call_response = {
            "content": [
                {"type": "text", "text": "Result line 1"},
                {"type": "text", "text": "Result line 2"},
            ]
        }

        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),  # /initialize
                _make_response(call_response),  # /tools/call
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("search", {"query": "test"})

        assert result == "Result line 1\nResult line 2"
        # Verify the call payload
        call_args = mock_http.post.call_args_list[1]
        assert call_args.kwargs["json"] == {"name": "search", "arguments": {"query": "test"}}

    @pytest.mark.asyncio
    async def test_sse_connection_error_graceful(self) -> None:
        """Server unreachable — list_tools returns empty list."""
        client = MCPSSEClient(base_url="http://localhost:9999")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),  # /initialize succeeds
                httpx.ConnectError("Connection refused"),  # /tools/list fails
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_sse_timeout(self) -> None:
        """Mock slow response, verify timeout handling."""
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),  # /initialize succeeds
                httpx.TimeoutException("Read timed out"),  # /tools/call times out
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("slow_tool", {"x": 1})

        assert "Error" in result
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_sse_disconnect(self) -> None:
        """Disconnect closes the HTTP client."""
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_make_response({}))
        mock_http.aclose = AsyncMock()

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            await client.disconnect()

        mock_http.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_sse_disconnect_when_not_connected(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_sse_connect_non_2xx_status(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_make_response({}, status_code=500))

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()

        mock_http.post.assert_called_once_with("/initialize", json={})

    @pytest.mark.asyncio
    async def test_sse_connect_connection_error_raises(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:9999")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(httpx.ConnectError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_sse_connect_timeout_raises(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:9999")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("connect timeout"))

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(httpx.TimeoutException):
                await client.connect()

    @pytest.mark.asyncio
    async def test_sse_call_tool_http_error(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),  # /initialize
                _make_response({}, status_code=500),  # /tools/call — 500 error
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("bad_tool", {})

        assert "Error" in result
        assert "call failed" in result

    @pytest.mark.asyncio
    async def test_sse_call_tool_connection_error(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                httpx.ConnectError("Connection refused"),
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("tool", {})

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_sse_call_tool_list_tools_timeout(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                httpx.TimeoutException("list_tools timed out"),
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_sse_call_tool_with_non_dict_content_item(self) -> None:
        call_response = {
            "content": [
                {"type": "text", "text": "Good result"},
                "just a string, not a dict",
                {"type": "text", "text": "Another result"},
                12345,
            ]
        }

        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                _make_response(call_response),
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("tool", {})

        assert "Good result" in result
        assert "Another result" in result

    @pytest.mark.asyncio
    async def test_sse_call_tool_with_empty_content_array(self) -> None:
        call_response = {"content": []}

        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                _make_response(call_response),
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("tool", {})

        assert "content" in result

    @pytest.mark.asyncio
    async def test_sse_call_tool_content_without_text_key(self) -> None:
        call_response = {
            "content": [
                {"type": "image", "data": "base64bytes"},
            ]
        }

        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                _make_response(call_response),
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("tool", {})

        assert "image" in result

    @pytest.mark.asyncio
    async def test_sse_response_body_too_large_without_header(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        big_content = b"x" * 2_000_000
        headers = {"content-type": "application/json"}
        big_resp = httpx.Response(
            status_code=200,
            content=big_content,
            headers=headers,
            request=httpx.Request("POST", "http://test"),
        )
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                big_resp,
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_sse_call_tool_invalid_json(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        bad_resp = httpx.Response(
            status_code=200,
            content=b"not valid json at all !!!",
            headers={"content-type": "text/plain"},
            request=httpx.Request("POST", "http://test"),
        )
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                bad_resp,
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("tool", {})

        assert "invalid JSON" in result

    @pytest.mark.asyncio
    async def test_sse_list_tools_invalid_json(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        bad_resp = httpx.Response(
            status_code=200,
            content=b"]]]broken json[[[",
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "http://test"),
        )
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                bad_resp,
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_sse_base_url_trailing_slash_stripped(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080/")
        assert client._base_url == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_sse_call_tool_response_too_large(self) -> None:
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        big_content = b"x" * 2_000_000
        headers = {}
        big_resp = httpx.Response(
            status_code=200,
            content=big_content,
            headers=headers,
            request=httpx.Request("POST", "http://test"),
        )
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),
                big_resp,
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            result = await client.call_tool("big_tool", {})

        assert "Error" in result

    def test_validate_response_size_body_too_large(self) -> None:
        from godspeed.mcp.sse_transport import MCPSSEClient

        big_content = b"x" * 2_000_000
        headers = {}
        resp = httpx.Response(
            status_code=200,
            content=big_content,
            headers=headers,
            request=httpx.Request("POST", "http://test"),
        )

        with pytest.raises(ValueError, match="too large"):
            MCPSSEClient._validate_response_size(resp)

    def test_validate_response_size_content_length_too_large(self) -> None:
        from godspeed.mcp.sse_transport import MCPSSEClient

        small_body = b'{"ok": true}'
        headers = {"content-length": str(2_000_000)}
        resp = httpx.Response(
            status_code=200,
            content=small_body,
            headers=headers,
            request=httpx.Request("POST", "http://test"),
        )

        with pytest.raises(ValueError, match="too large"):
            MCPSSEClient._validate_response_size(resp)

    @pytest.mark.asyncio
    async def test_sse_call_tool_not_connected(self) -> None:
        """Call tool before connecting returns error string."""
        client = MCPSSEClient(base_url="http://localhost:8080")
        result = await client.call_tool("anything", {})
        assert "Error" in result
        assert "not connected" in result

    @pytest.mark.asyncio
    async def test_sse_list_tools_not_connected(self) -> None:
        """List tools before connecting returns empty list."""
        client = MCPSSEClient(base_url="http://localhost:8080")
        tools = await client.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_sse_response_size_validation(self) -> None:
        """Oversized response is rejected."""
        client = MCPSSEClient(base_url="http://localhost:8080")
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        oversized_resp = _make_response({}, content_length=2_000_000)
        mock_http.post = AsyncMock(
            side_effect=[
                _make_response({}),  # /initialize
                oversized_resp,  # /tools/list — too large
            ]
        )

        with patch("godspeed.mcp.sse_transport.httpx.AsyncClient", return_value=mock_http):
            await client.connect()
            tools = await client.list_tools()

        assert tools == []


class TestMCPTransportSelection:
    """Tests for transport selection in MCPClient."""

    def test_transport_selection_stdio(self) -> None:
        """Config with transport=stdio uses existing client path."""
        config = MCPServerConfig(name="local", command="echo", transport="stdio")
        assert config.transport == "stdio"
        assert config.url is None

    def test_transport_selection_sse(self) -> None:
        """Config with transport=sse populates url and headers."""
        config = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:8080",
            headers={"Authorization": "Bearer tok"},
        )
        assert config.transport == "sse"
        assert config.url == "http://localhost:8080"
        assert config.headers == {"Authorization": "Bearer tok"}

    def test_config_backward_compat(self) -> None:
        """Config without transport field defaults to stdio."""
        config = MCPServerConfig(name="legacy", command="node")
        assert config.transport == "stdio"
        assert config.url is None
        assert config.headers is None

    @pytest.mark.asyncio
    async def test_connect_routes_to_sse(self) -> None:
        """MCPClient.connect dispatches to SSE path for transport=sse."""
        client = MCPClient()
        config = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:8080",
        )

        mock_sse = AsyncMock()
        mock_sse.connect = AsyncMock()
        mock_sse.list_tools = AsyncMock(
            return_value=[
                {
                    "name": "ping",
                    "description": "Ping test",
                    "inputSchema": {},
                }
            ]
        )

        with patch("godspeed.mcp.sse_transport.MCPSSEClient", return_value=mock_sse):
            tools = await client.connect(config)

        assert len(tools) == 1
        assert tools[0].name == "mcp_remote_ping"
        mock_sse.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_sse_missing_url(self) -> None:
        """SSE transport without url returns empty list."""
        client = MCPClient()
        config = MCPServerConfig(name="broken", transport="sse")
        tools = await client.connect(config)
        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_routes_to_sse(self) -> None:
        """MCPClient.call_tool dispatches to SSE client when available."""
        client = MCPClient()
        mock_sse = AsyncMock()
        mock_sse.call_tool = AsyncMock(return_value="pong")
        client._sse_clients["remote"] = mock_sse

        result = await client.call_tool("remote", "ping", {})
        assert result == "pong"
        mock_sse.call_tool.assert_called_once_with("ping", {})

    @pytest.mark.asyncio
    async def test_disconnect_all_closes_sse(self) -> None:
        """disconnect_all closes SSE clients."""
        client = MCPClient()
        mock_sse = AsyncMock()
        mock_sse.disconnect = AsyncMock()
        client._sse_clients["remote"] = mock_sse

        await client.disconnect_all()

        mock_sse.disconnect.assert_called_once()
        assert len(client._sse_clients) == 0
        assert len(client._connections) == 0
