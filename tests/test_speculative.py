"""Tests for speculative tool execution during streaming (Unit 15)."""

from __future__ import annotations

import asyncio

import pytest

from godspeed.agent.loop import _speculative_dispatch
from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ReadTool(Tool):
    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "read"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments, context):
        return ToolResult.success("read content")


class _WriteTool(Tool):
    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "write"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments, context):
        return ToolResult.success("written")


# ---------------------------------------------------------------------------
# _speculative_dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speculative_dispatches_read_only(tmp_path):
    """READ_ONLY tools get dispatched into the cache."""
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(_WriteTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "tc1", "function": {"name": "file_read", "arguments": "{}"}},
        {"id": "tc2", "function": {"name": "file_write", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)

    # Only read tool should be cached
    assert "tc1" in cache
    assert "tc2" not in cache

    # Await the cached task
    result = await cache["tc1"]
    assert result.output == "read content"


@pytest.mark.asyncio
async def test_speculative_skips_unknown_tools(tmp_path):
    """Unknown tools are not dispatched."""
    registry = ToolRegistry()
    registry.register(_ReadTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "tc1", "function": {"name": "unknown_tool", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_speculative_skips_malformed_calls(tmp_path):
    """Malformed tool calls are skipped."""
    registry = ToolRegistry()
    registry.register(_ReadTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "tc1", "function": {"name": "", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_speculative_no_duplicate_dispatch(tmp_path):
    """Same call_id is not dispatched twice."""
    registry = ToolRegistry()
    registry.register(_ReadTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "tc1", "function": {"name": "file_read", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)
    first_task = cache["tc1"]

    # Dispatch again with same id
    _speculative_dispatch(raw_calls, registry, ctx, cache)
    assert cache["tc1"] is first_task  # Same task, not replaced

    await first_task


@pytest.mark.asyncio
async def test_speculative_empty_call_id_skipped(tmp_path):
    """Tool calls with empty call_id are not speculatively dispatched."""
    registry = ToolRegistry()
    registry.register(_ReadTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "", "function": {"name": "file_read", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_speculative_cache_consumed_by_loop(tmp_path):
    """Speculative cache entries are consumed (popped) when used."""
    registry = ToolRegistry()
    registry.register(_ReadTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "tc1", "function": {"name": "file_read", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)
    assert "tc1" in cache

    # Simulate main loop consuming the cache
    task = cache.pop("tc1")
    result = await task
    assert result.output == "read content"
    assert "tc1" not in cache


@pytest.mark.asyncio
async def test_speculative_multiple_read_tools(tmp_path):
    """Multiple READ_ONLY calls are all dispatched."""

    class _GrepTool(Tool):
        @property
        def name(self) -> str:
            return "grep_search"

        @property
        def description(self) -> str:
            return "grep"

        @property
        def risk_level(self) -> RiskLevel:
            return RiskLevel.READ_ONLY

        def get_schema(self):
            return {"type": "object", "properties": {}}

        async def execute(self, arguments, context):
            return ToolResult.success("grep results")

    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(_GrepTool())
    registry.register(_WriteTool())

    ctx = ToolContext(cwd=tmp_path, session_id="test")
    cache: dict[str, asyncio.Task[ToolResult]] = {}

    raw_calls = [
        {"id": "tc1", "function": {"name": "file_read", "arguments": "{}"}},
        {"id": "tc2", "function": {"name": "grep_search", "arguments": "{}"}},
        {"id": "tc3", "function": {"name": "file_write", "arguments": "{}"}},
    ]

    _speculative_dispatch(raw_calls, registry, ctx, cache)
    assert "tc1" in cache
    assert "tc2" in cache
    assert "tc3" not in cache

    r1 = await cache["tc1"]
    r2 = await cache["tc2"]
    assert r1.output == "read content"
    assert r2.output == "grep results"
