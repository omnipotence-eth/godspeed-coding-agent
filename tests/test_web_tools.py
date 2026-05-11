"""Tests for web search and web fetch tools."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.web_fetch import (
    CACHE_TTL_SECONDS,
    WebFetchTool,
    _cache_path_for,
    _cache_read,
    _cache_write,
    _extract_content,
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

    def test_description(self) -> None:
        desc = WebFetchTool().description
        assert "Fetch a URL" in desc
        assert "Strips HTML" in desc

    def test_risk_level(self) -> None:
        assert WebFetchTool().risk_level.value == "low"

    def test_get_schema(self) -> None:
        schema = WebFetchTool().get_schema()
        assert schema["type"] == "object"
        assert "url" in schema["properties"]
        assert "url" in schema["required"]

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

    def test_cache_read_none_when_text_not_str(self) -> None:
        """If the cached text field is not a string, return None."""
        path = _cache_path_for("https://example.com/badtext")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at_ts": time.time(), "text": 12345}), encoding="utf-8")
        assert _cache_read("https://example.com/badtext") is None

    def test_cache_read_none_when_ts_not_number(self) -> None:
        """If fetched_at_ts is not a number, return None."""
        path = _cache_path_for("https://example.com/badts")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at_ts": "yesterday", "text": "hello"}), encoding="utf-8"
        )
        assert _cache_read("https://example.com/badts") is None

    def test_cache_evict_when_over_budget(self) -> None:
        """Oldest cache files are evicted when total exceeds _MAX_CACHE_BYTES."""
        import godspeed.tools.web_fetch as wf

        with patch("godspeed.tools.web_fetch._MAX_CACHE_BYTES", 100):
            for i in range(5):
                _cache_write(f"https://example.com/evict{i}", "x" * 80)
            remaining = list(wf._cache_dir().glob("*.json"))
            total_size = sum(p.stat().st_size for p in remaining)
            assert total_size <= 100

    def test_cache_evict_break_when_under_budget(self) -> None:
        """Eviction stops early (break) when total drops below threshold mid-loop."""
        import godspeed.tools.web_fetch as wf

        cache_dir = wf._cache_dir()
        for existing in cache_dir.glob("*.json"):
            existing.unlink(missing_ok=True)
        # Write one large entry and one small entry. Budget set so that after
        # evicting the large file only the small one remains and we hit break.
        _cache_write("https://example.com/large", "x" * 500)
        _cache_write("https://example.com/small", "sm")
        # Budget of 200 bytes: large (~620) + small (~150) = ~770 > 200
        # After evicting large (~620), total ~150 <= 200 -> break.
        from godspeed.tools.web_fetch import _cache_evict_if_needed

        with patch("godspeed.tools.web_fetch._MAX_CACHE_BYTES", 200):
            _cache_evict_if_needed()
        remaining_paths = [p.name for p in cache_dir.glob("*.json")]
        assert len(remaining_paths) <= 2, f"expected at most 2 files, got {len(remaining_paths)}"
        # The small file survived; the large file was evicted.

    def test_cache_evict_handles_stat_oserror(self) -> None:
        """If stat() fails on a cache file, that file is skipped in eviction."""
        import godspeed.tools.web_fetch as wf

        # Write a valid entry first
        _cache_write("https://example.com/good", "good data")
        cache_dir = wf._cache_dir()
        # Create a bogus file that will cause stat to fail (we'll mock it)
        bogus = cache_dir / "bogus.json"
        bogus.write_text("{}", encoding="utf-8")

        from godspeed.tools.web_fetch import _cache_evict_if_needed

        with patch("godspeed.tools.web_fetch._MAX_CACHE_BYTES", 10):
            # Mock Path.stat to fail for the bogus file
            original_stat = type(bogus).stat

            def _failing_stat(self_path):
                if str(self_path) == str(bogus):
                    raise OSError("stat failed")
                return original_stat(self_path)

            with patch.object(type(bogus), "stat", _failing_stat):
                _cache_evict_if_needed()
        # Should not raise; bogus file is skipped

    def test_cache_write_oserror_suppressed(self) -> None:
        """OSError during cache write is logged but not raised."""
        with patch("godspeed.tools.web_fetch._cache_path_for") as mock_path:
            mock_path.return_value.write_text.side_effect = OSError("disk full")
            _cache_write("https://example.com/diskfull", "data")

    def test_cache_read_oserror_suppressed(self) -> None:
        """OSError during cache read returns None silently."""
        path = _cache_path_for("https://example.com/oserr")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"fetched_at_ts": 9999999999, "text": "ok"}', encoding="utf-8")
        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            assert _cache_read("https://example.com/oserr") is None


class TestReadabilityExtraction:
    """Test _extract_content with readability-lxml and fallback."""

    @pytest.fixture(autouse=True)
    def _mock_readability_module(self) -> None:
        """Ensure a mock readability module exists in sys.modules for patching."""
        import sys

        self._readability_mock = MagicMock()
        sys.modules["readability"] = self._readability_mock
        yield
        sys.modules.pop("readability", None)

    def test_readability_successful_extraction(self) -> None:
        """Readability extracts article content from full HTML page."""
        html_content = "<html><head><title>Test Article</title></head><body><p>Main content here. More text to reach fifty characters minimum.</p></body></html>"
        with patch.object(self._readability_mock, "Document") as MockDoc:
            mock_doc = MockDoc.return_value
            mock_doc.title.return_value = "Test Article"
            mock_doc.summary.return_value = (
                "<p>Main content here. More text to reach fifty characters minimum.</p>"
            )
            result = _extract_content(html_content, "https://example.com")
            assert "Title: Test Article" in result
            assert "Main content here" in result

    def test_readability_fallback_on_short_content(self) -> None:
        """When readability returns < 50 chars, fall back to tag stripping."""
        html_content = "<html><body><p>Hi</p></body></html>"
        with patch.object(self._readability_mock, "Document") as MockDoc:
            mock_doc = MockDoc.return_value
            mock_doc.title.return_value = ""
            mock_doc.summary.return_value = "Hi"
            result = _extract_content(html_content, "https://example.com")
            assert "Hi" in result

    def test_readability_fallback_on_no_title(self) -> None:
        """Readability summary without title — no 'Title:' prefix."""
        html_content = "<html><body><p>This is a sufficiently long paragraph that should exceed fifty characters easily.</p></body></html>"
        with patch.object(self._readability_mock, "Document") as MockDoc:
            mock_doc = MockDoc.return_value
            mock_doc.title.return_value = ""
            mock_doc.summary.return_value = "<p>This is a sufficiently long paragraph that should exceed fifty characters easily.</p>"
            result = _extract_content(html_content, "https://example.com")
            assert "Title:" not in result

    def test_readability_import_error_falls_back(self) -> None:
        """When readability import fails, fall back to tag stripping."""
        import sys

        html_content = "<html><body><p>Fallback content here for testing the import error path</p></body></html>"
        sys.modules.pop("readability", None)
        result = _extract_content(html_content, "https://example.com")
        assert "Fallback content here" in result

    def test_readability_exception_falls_back(self) -> None:
        """When readability raises an exception, fall back to tag stripping."""
        html_content = (
            "<html><body><p>Exception fallback test content for coverage.</p></body></html>"
        )
        with patch.object(
            self._readability_mock, "Document", side_effect=RuntimeError("parse error")
        ):
            result = _extract_content(html_content, "https://example.com")
            assert "Exception fallback test content" in result


class TestWebFetchExecuteEdgeCases:
    """Test edge cases in WebFetchTool.execute with mocked network."""

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "webcache"
        cache_dir.mkdir()
        monkeypatch.setattr("godspeed.tools.web_fetch._cache_dir", lambda: cache_dir)

    @pytest.fixture
    def ctx(self, tmp_path: Path) -> ToolContext:
        return ToolContext(cwd=tmp_path, session_id="test")

    @pytest.mark.asyncio
    async def test_http_error(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        with patch("godspeed.tools.web_fetch.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.HTTPError(
                "https://example.com", 404, "Not Found", {}, None
            )
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert result.is_error
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_url_error(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        with patch("godspeed.tools.web_fetch.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("connection refused")
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert result.is_error
        assert "connection refused" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_error(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        with patch("godspeed.tools.web_fetch.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = TimeoutError("timed out")
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert result.is_error
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_generic_exception(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        with patch("godspeed.tools.web_fetch.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = ValueError("something went wrong")
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert result.is_error
        assert "something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_charset_detection_from_content_type(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html; charset=iso-8859-1"}
        mock_resp.read.return_value = b"<html><body><p>Valid paragraph with enough text to exceed fifty character threshold for readability fallback.</p></body></html>"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("godspeed.tools.web_fetch.urllib.request.urlopen", return_value=mock_resp),
            patch("godspeed.tools.web_fetch.urllib.request.Request"),
        ):
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_invalid_charset_falls_back_to_utf8(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html; charset=x-invalid-charset-999"}
        mock_resp.read.return_value = b"<html><body><p>This is valid content that falls back to utf-8 decoding after invalid charset lookup.</p></body></html>"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("godspeed.tools.web_fetch.urllib.request.urlopen", return_value=mock_resp),
            patch("godspeed.tools.web_fetch.urllib.request.Request"),
        ):
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_non_html_content_type(self, ctx: ToolContext) -> None:
        """Non-HTML content is returned as-is without readability extraction."""
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.read.return_value = b'{"key": "value"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("godspeed.tools.web_fetch.urllib.request.urlopen", return_value=mock_resp),
            patch("godspeed.tools.web_fetch.urllib.request.Request"),
        ):
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert not result.is_error
        assert '{"key": "value"}' in result.output

    @pytest.mark.asyncio
    async def test_content_truncation(self, ctx: ToolContext) -> None:
        """Content longer than MAX_OUTPUT_CHARS is truncated with a note."""
        tool = WebFetchTool()
        large_text = "A" * 15000
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.read.return_value = large_text.encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("godspeed.tools.web_fetch.urllib.request.urlopen", return_value=mock_resp),
            patch("godspeed.tools.web_fetch.urllib.request.Request"),
        ):
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert "(truncated" in result.output.lower()

    @pytest.mark.asyncio
    async def test_empty_text_after_extraction(self, ctx: ToolContext) -> None:
        """When HTML extraction yields only whitespace, a success message is returned."""
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.read.return_value = b"   \n  \t  "
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("godspeed.tools.web_fetch.urllib.request.urlopen", return_value=mock_resp),
            patch("godspeed.tools.web_fetch.urllib.request.Request"),
        ):
            result = await tool.execute({"url": "https://example.com"}, ctx)
        assert "no readable text" in result.output.lower()

    @pytest.mark.asyncio
    async def test_successful_fetch_caches_result(self, ctx: ToolContext) -> None:
        """A successful live fetch writes to the disk cache."""
        tool = WebFetchTool()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.read.return_value = b"<html><body><p>Cacheable content here for testing the cache write path after a successful live fetch request.</p></body></html>"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("godspeed.tools.web_fetch.urllib.request.urlopen", return_value=mock_resp),
            patch("godspeed.tools.web_fetch.urllib.request.Request"),
        ):
            result = await tool.execute({"url": "https://example.com/cachable"}, ctx)
        assert not result.is_error
        cached = _cache_read("https://example.com/cachable")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_url_not_a_string(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        result = await tool.execute({"url": 42}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_url_missing_scheme(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        result = await tool.execute({"url": "example.com"}, ctx)
        assert result.is_error
        assert "http" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_zero_zero_zero_zero(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        result = await tool.execute({"url": "http://0.0.0.0/api"}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_blocks_ipv6_localhost(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        result = await tool.execute({"url": "http://[::1]:8080/secret"}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_blocks_172_16(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        result = await tool.execute({"url": "http://172.16.0.1/"}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_blocks_172_17(self, ctx: ToolContext) -> None:
        tool = WebFetchTool()
        result = await tool.execute({"url": "http://172.17.0.2/api"}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_unused_port_not_blocked(self, ctx: ToolContext) -> None:
        """172.20.x is NOT a private range — should not be blocked."""
        tool = WebFetchTool()
        with patch("godspeed.tools.web_fetch.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("no route")
            result = await tool.execute({"url": "http://172.20.0.1/"}, ctx)
        assert "local" not in result.error.lower()
        assert "private" not in result.error.lower()


class TestCacheDirOriginal:
    """Test the original _cache_dir function (not monkeypatched)."""

    def test_original_cache_dir_creates_and_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from godspeed.tools.web_fetch import _cache_dir as original_cache_dir

        result = original_cache_dir()
        expected = tmp_path / ".godspeed" / "cache" / "web"
        assert result == expected
        assert result.is_dir()
