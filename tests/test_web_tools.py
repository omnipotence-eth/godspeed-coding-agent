"""Tests for web search and web fetch tools."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.web_fetch import (
    CACHE_TTL_SECONDS,
    WebFetchTool,
    _cache_path_for,
    _cache_read,
    _cache_write,
    _html_to_text,
    _is_local_url,
)
from godspeed.tools.web_search import WebSearchTool, _parse_ddg_html


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test")


class TestHtmlToText:
    """Test HTML-to-text extraction."""

    def test_strips_tags(self) -> None:
        result = _html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_strips_script_tags(self) -> None:
        result = _html_to_text("<script>alert('xss')</script>Content here")
        assert "alert" not in result
        assert "Content here" in result

    def test_strips_style_tags(self) -> None:
        result = _html_to_text("<style>body{color:red}</style>Visible")
        assert "color" not in result
        assert "Visible" in result

    def test_decodes_entities(self) -> None:
        result = _html_to_text("&amp; &lt; &gt; &quot;")
        assert "&" in result
        assert "<" in result

    def test_empty_input(self) -> None:
        assert _html_to_text("") == ""

    def test_plain_text_passthrough(self) -> None:
        assert _html_to_text("plain text") == "plain text"


class TestIsLocalUrl:
    """Test local URL detection."""

    def test_localhost(self) -> None:
        assert _is_local_url("http://localhost:8000/api")

    def test_127_0_0_1(self) -> None:
        assert _is_local_url("http://127.0.0.1:3000")

    def test_private_192(self) -> None:
        assert _is_local_url("http://192.168.1.100/")

    def test_private_10(self) -> None:
        assert _is_local_url("http://10.0.0.1/api")

    def test_public_url_not_local(self) -> None:
        assert not _is_local_url("https://example.com")

    def test_github_not_local(self) -> None:
        assert not _is_local_url("https://github.com/user/repo")


class TestWebFetchTool:
    """Test web fetch tool."""

    def test_name(self) -> None:
        assert WebFetchTool().name == "web_fetch"

    @pytest.mark.asyncio
    async def test_empty_url(self, ctx: ToolContext) -> None:
        result = await WebFetchTool().execute({"url": ""}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_invalid_protocol(self, ctx: ToolContext) -> None:
        result = await WebFetchTool().execute({"url": "ftp://example.com"}, ctx)
        assert result.is_error
        assert "http" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_localhost(self, ctx: ToolContext) -> None:
        result = await WebFetchTool().execute({"url": "http://localhost:8000/secret"}, ctx)
        assert result.is_error
        assert "local" in result.error.lower() or "private" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_private_ip(self, ctx: ToolContext) -> None:
        result = await WebFetchTool().execute({"url": "http://192.168.1.1/admin"}, ctx)
        assert result.is_error


class TestWebSearchTool:
    """Test web search tool."""

    def test_name(self) -> None:
        assert WebSearchTool().name == "web_search"

    @pytest.mark.asyncio
    async def test_empty_query(self, ctx: ToolContext) -> None:
        result = await WebSearchTool().execute({"query": ""}, ctx)
        assert result.is_error

    def test_schema_has_query(self) -> None:
        schema = WebSearchTool().get_schema()
        assert "query" in schema["properties"]
        assert "query" in schema["required"]


class TestParseDdgHtml:
    """Test DuckDuckGo HTML parsing."""

    def test_parses_result_links(self) -> None:
        html = """
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <a class="result__snippet">This is a snippet</a>
        </div>
        """
        results = _parse_ddg_html(html, max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "This is a snippet"

    def test_empty_html_returns_empty(self) -> None:
        results = _parse_ddg_html("", max_results=5)
        assert results == []

    def test_respects_max_results(self) -> None:
        html = ""
        for i in range(10):
            html += f'<a class="result__a" href="https://example.com/{i}">Title {i}</a>'
        results = _parse_ddg_html(html, max_results=3)
        assert len(results) == 3


class TestWebFetchCache:
    """Test the 7-day disk cache layer on WebFetchTool.

    All tests redirect the cache dir into a tmp_path so we never touch
    the real ``~/.godspeed/cache/web/`` during CI.
    """

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redirect _cache_dir() to a tmp path so tests don't pollute real cache."""
        cache_dir = tmp_path / "webcache"
        cache_dir.mkdir()
        monkeypatch.setattr("godspeed.tools.web_fetch._cache_dir", lambda: cache_dir)

    def test_cache_miss_returns_none(self) -> None:
        assert _cache_read("https://example.com/never-seen") is None

    def test_cache_write_then_read(self) -> None:
        _cache_write("https://example.com/foo", "cached body text")
        got = _cache_read("https://example.com/foo")
        assert got == "cached body text"

    def test_cache_expired_returns_none(self) -> None:
        """An entry older than TTL must be treated as a miss."""
        _cache_write("https://example.com/stale", "old text")
        # Monkeypatch time.time to fast-forward past TTL
        with patch(
            "godspeed.tools.web_fetch.time.time", return_value=time.time() + CACHE_TTL_SECONDS + 1
        ):
            got = _cache_read("https://example.com/stale")
        assert got is None

    def test_cache_hit_shortcircuits_execute(self, ctx: ToolContext) -> None:
        """A cached URL must return cached content without hitting the network."""

        _cache_write("https://example.com/doc", "pre-cached body")
        # Patch urlopen to explode if called — proves the cache short-circuited.
        with patch(
            "godspeed.tools.web_fetch.urllib.request.urlopen",
            side_effect=AssertionError("urlopen should not be called on a cache hit"),
        ):
            result = asyncio.run(WebFetchTool().execute({"url": "https://example.com/doc"}, ctx))
        assert result.is_error is False
        assert "pre-cached body" in result.output
        assert "(cached)" in result.output

    def test_no_cache_flag_bypasses_cache(self, ctx: ToolContext) -> None:
        """Passing no_cache=True must force a live fetch even with a fresh entry."""
        import contextlib

        _cache_write("https://example.com/live", "stale cache body")
        with (
            patch(
                "godspeed.tools.web_fetch.urllib.request.urlopen",
                side_effect=AssertionError("network call with no_cache=True — expected"),
            ) as mock_urlopen,
            contextlib.suppress(AssertionError),
        ):
            # Urlopen is called (proving cache bypass) and raises AssertionError;
            # we swallow it to keep the test focused on the bypass decision.
            asyncio.run(
                WebFetchTool().execute({"url": "https://example.com/live", "no_cache": True}, ctx)
            )
        # The assertion error inside urlopen proves the cache was bypassed.
        mock_urlopen.assert_called_once()

    def test_cache_survives_corrupt_file(self, tmp_path: Path) -> None:
        """A corrupt cache file must not crash the reader."""

        path = _cache_path_for("https://example.com/corrupt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        # _cache_read must return None, not raise.
        assert _cache_read("https://example.com/corrupt") is None
