"""MCP SSE/HTTP transport — connect to remote MCP servers over HTTP."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Safety limits
_TIMEOUT_SECONDS = 30.0
_MAX_RESPONSE_BYTES = 1_048_576  # 1 MB


class MCPSSEClient:
    """MCP client for SSE/HTTP transport (remote MCP servers).

    Uses JSON-RPC over HTTP — POST requests to well-known endpoints.
    Does NOT depend on httpx-sse; responses are plain JSON.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize connection to SSE MCP server."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=httpx.Timeout(_TIMEOUT_SECONDS),
        )
        try:
            resp = await self._client.post("/initialize", json={})
            resp.raise_for_status()
            logger.info("MCP SSE connected url=%s", self._base_url)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "MCP SSE initialize returned non-2xx url=%s status=%d",
                self._base_url,
                exc.response.status_code,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.error("MCP SSE connect failed url=%s error=%s", self._base_url, exc)
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the server.

        Returns:
            List of tool definition dicts with keys: name, description, inputSchema.
        """
        if self._client is None:
            logger.error("MCP SSE client not connected — call connect() first")
            return []

        try:
            resp = await self._client.post("/tools/list", json={})
            resp.raise_for_status()
            self._validate_response_size(resp)
            data = resp.json()
            tools: list[dict[str, Any]] = data.get("tools", [])
            return tools
        except httpx.TimeoutException:
            logger.error("MCP SSE list_tools timed out url=%s", self._base_url)
            return []
        except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.error("MCP SSE list_tools failed url=%s error=%s", self._base_url, exc)
            return []
        except ValueError:
            logger.error("MCP SSE list_tools returned invalid JSON url=%s", self._base_url)
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the remote server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments dict.

        Returns:
            The tool result as a string.
        """
        if self._client is None:
            return "Error: MCP SSE client not connected"

        try:
            resp = await self._client.post(
                "/tools/call",
                json={"name": tool_name, "arguments": arguments},
            )
            resp.raise_for_status()
            self._validate_response_size(resp)
            data = resp.json()

            # Extract text from content array (MCP standard format)
            content = data.get("content", [])
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            return "\n".join(parts) if parts else str(data)
        except httpx.TimeoutException:
            logger.error(
                "MCP SSE call_tool timed out url=%s tool=%s",
                self._base_url,
                tool_name,
            )
            return f"Error: MCP SSE call timed out for tool '{tool_name}'"
        except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.error(
                "MCP SSE call_tool failed url=%s tool=%s error=%s",
                self._base_url,
                tool_name,
                exc,
            )
            return f"Error: MCP SSE call failed for tool '{tool_name}' — {exc}"
        except ValueError:
            logger.error(
                "MCP SSE call_tool returned invalid JSON url=%s tool=%s",
                self._base_url,
                tool_name,
            )
            return f"Error: MCP SSE returned invalid JSON for tool '{tool_name}'"

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("MCP SSE disconnected url=%s", self._base_url)

    @staticmethod
    def _validate_response_size(resp: httpx.Response) -> None:
        """Reject responses exceeding the safety limit."""
        content_length = resp.headers.get("content-length")
        if content_length is not None and int(content_length) > _MAX_RESPONSE_BYTES:
            msg = (
                f"MCP SSE response too large: {content_length} bytes (limit {_MAX_RESPONSE_BYTES})"
            )
            raise ValueError(msg)
        if len(resp.content) > _MAX_RESPONSE_BYTES:
            msg = (
                f"MCP SSE response too large: {len(resp.content)} bytes "
                f"(limit {_MAX_RESPONSE_BYTES})"
            )
            raise ValueError(msg)
