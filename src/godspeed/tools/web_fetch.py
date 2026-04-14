"""Web fetch tool — retrieve web pages and extract readable text.

Uses urllib (stdlib) for zero-dependency HTTP, with a simple HTML tag stripper
for readable output. No external dependencies required.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Max content size to download (1MB)
MAX_CONTENT_BYTES = 1_000_000
# Max text output to return
MAX_OUTPUT_CHARS = 10_000
# Request timeout
FETCH_TIMEOUT = 15

# Tags whose content should be stripped entirely
_STRIP_TAGS = re.compile(
    r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# All HTML tags
_HTML_TAGS = re.compile(r"<[^>]+>")
# Collapse whitespace
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _html_to_text(raw_html: str) -> str:
    """Extract readable text from HTML — simple tag stripping."""
    # Remove script/style/head blocks
    text = _STRIP_TAGS.sub("", raw_html)
    # Remove all remaining tags
    text = _HTML_TAGS.sub("\n", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


class WebFetchTool(Tool):
    """Fetch a web page and return readable text content.

    Strips HTML tags, scripts, and styles to return clean text.
    Useful for reading documentation, error messages, and API references.
    """

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL and return the page content as readable text. "
            "Strips HTML to plain text. Use for documentation, API refs, error lookups. "
            "Max 10K chars returned.\n\n"
            "Example: web_fetch(url='https://docs.python.org/3/library/asyncio.html')\n"
            "Example: web_fetch(url='https://fastapi.tiangolo.com/tutorial/dependencies/')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must start with http:// or https://)",
                },
            },
            "required": ["url"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        url = arguments.get("url", "")

        if not isinstance(url, str) or not url:
            return ToolResult.failure("url must be a non-empty string")

        if not url.startswith(("http://", "https://")):
            return ToolResult.failure("url must start with http:// or https://")

        # Block local/private network access
        if _is_local_url(url):
            return ToolResult.failure("Cannot fetch local/private network URLs")

        try:
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={"User-Agent": "Godspeed-Agent/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:  # noqa: S310  # nosec B310
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read(MAX_CONTENT_BYTES)

        except urllib.error.HTTPError as exc:
            return ToolResult.failure(f"HTTP {exc.code}: {exc.reason}")
        except urllib.error.URLError as exc:
            return ToolResult.failure(f"Failed to fetch URL: {exc.reason}")
        except TimeoutError:
            return ToolResult.failure(f"Request timed out after {FETCH_TIMEOUT}s")
        except Exception as exc:
            return ToolResult.failure(f"Fetch error: {exc}")

        # Decode
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", errors="replace")

        # Extract text from HTML
        if "html" in content_type.lower():
            text = _html_to_text(text)

        # Truncate
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n\n... (truncated, {len(text)} total chars)"

        if not text.strip():
            return ToolResult.success(f"Fetched {url} but no readable text content found.")

        return ToolResult.success(f"Content from {url}:\n\n{text}")


def _is_local_url(url: str) -> bool:
    """Check if a URL points to local/private network addresses."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or hostname.startswith(  # noqa: S104  # nosec B104
        ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.")
    )
