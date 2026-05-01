"""Tests for retry logic — connection errors, transient failures, backoff math.

Covers LLMClient connection classification and ToolRegistry transient-retry
loops that were added in the reliability audit but lacked dedicated tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.llm.client import LLMClient
from godspeed.tools.base import Tool, ToolCall, ToolContext, ToolResult
from godspeed.tools.registry import ToolRegistry, _is_transient_error

# ---------------------------------------------------------------------------
# LLMClient: connection-error classification
# ---------------------------------------------------------------------------


class TestIsConnectionError:
    def test_connection_refused(self) -> None:
        c = LLMClient(model="test")
        assert c._is_connection_error(Exception("connection refused"))

    def test_cannot_connect(self) -> None:
        c = LLMClient(model="test")
        assert c._is_connection_error(Exception("cannot connect to host"))

    def test_connect_call_failed(self) -> None:
        c = LLMClient(model="test")
        assert c._is_connection_error(Exception("connect call failed"))

    def test_random_error_not_connection(self) -> None:
        c = LLMClient(model="test")
        assert not c._is_connection_error(Exception("something went wrong"))

    def test_timeout_not_connection(self) -> None:
        c = LLMClient(model="test")
        assert not c._is_connection_error(Exception("request timed out"))


# ---------------------------------------------------------------------------
# LLMClient: backoff-delay math
# ---------------------------------------------------------------------------


class TestBackoffDelay:
    def test_exponential_without_retry_after(self) -> None:
        c = LLMClient(model="test")
        d0 = c._backoff_delay(0, None)
        d1 = c._backoff_delay(1, None)
        d2 = c._backoff_delay(2, None)
        # Base doubles each time (1, 2, 4) with ±25% jitter
        assert 0.75 <= d0 <= 1.25
        assert 1.50 <= d1 <= 2.50
        assert 3.00 <= d2 <= 5.00

    def test_retry_after_used_as_floor(self) -> None:
        c = LLMClient(model="test")
        d = c._backoff_delay(0, 10.0)
        # Must be >= 10.0 (upward-only jitter)
        assert d >= 10.0

    def test_retry_after_clamped_to_max(self) -> None:
        c = LLMClient(model="test")
        d = c._backoff_delay(0, 99999.0)
        assert d <= 60.0

    def test_backoff_never_negative(self) -> None:
        c = LLMClient(model="test")
        for i in range(10):
            d = c._backoff_delay(i, None)
            assert d >= 0.0


# ---------------------------------------------------------------------------
# ToolRegistry: transient-error classification
# ---------------------------------------------------------------------------


class TestIsTransientError:
    def test_timeout(self) -> None:
        assert _is_transient_error("Request timeout")

    def test_connection_refused(self) -> None:
        assert _is_transient_error("connection refused")

    def test_connection_reset(self) -> None:
        assert _is_transient_error("Connection reset by peer")

    def test_rate_limit(self) -> None:
        assert _is_transient_error("rate limit exceeded")

    def test_502_503_504(self) -> None:
        assert _is_transient_error("502 Bad Gateway")
        assert _is_transient_error("503 Service Unavailable")
        assert _is_transient_error("504 Gateway Timeout")

    def test_permission_denial_not_transient(self) -> None:
        assert not _is_transient_error("Permission denied")

    def test_file_not_found_not_transient(self) -> None:
        assert not _is_transient_error("File not found")

    def test_logic_error_not_transient(self) -> None:
        assert not _is_transient_error("division by zero")


# ---------------------------------------------------------------------------
# ToolRegistry: dispatch retry loop
# ---------------------------------------------------------------------------


class _FlakyTool(Tool):
    """Tool that fails N times with a transient error, then succeeds."""

    def __init__(self, fail_count: int = 0) -> None:
        self.fail_count = fail_count
        self.attempts = 0

    @property
    def name(self) -> str:
        return "flaky"

    @property
    def description(self) -> str:
        return "A flaky tool for testing retries"

    @property
    def risk_level(self) -> str:
        return "LOW"

    def get_schema(self) -> dict:
        return {}

    async def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise TimeoutError("connection reset")
        return ToolResult.success("ok")


class _PermanentFailTool(Tool):
    """Tool that always fails with a non-transient error."""

    @property
    def name(self) -> str:
        return "permanent"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def risk_level(self) -> str:
        return "LOW"

    def get_schema(self) -> dict:
        return {}

    async def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        raise ValueError("bad argument")


class TestToolRegistryRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        registry = ToolRegistry(max_retries=2)
        tool = _FlakyTool(fail_count=0)
        registry.register(tool)

        result = await registry.dispatch(
            ToolCall(tool_name="flaky", arguments={}), ToolContext(cwd=Path("."), session_id="test")
        )
        assert not result.is_error
        assert tool.attempts == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        registry = ToolRegistry(max_retries=3)
        tool = _FlakyTool(fail_count=2)
        registry.register(tool)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await registry.dispatch(
                ToolCall(tool_name="flaky", arguments={}),
                ToolContext(cwd=Path("."), session_id="test"),
            )
        assert not result.is_error
        assert tool.attempts == 3

    @pytest.mark.asyncio
    async def test_retries_exhausted_then_fails(self) -> None:
        registry = ToolRegistry(max_retries=2)
        tool = _FlakyTool(fail_count=5)
        registry.register(tool)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await registry.dispatch(
                ToolCall(tool_name="flaky", arguments={}),
                ToolContext(cwd=Path("."), session_id="test"),
            )
        assert result.is_error
        assert tool.attempts == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_transient_error_no_retry(self) -> None:
        registry = ToolRegistry(max_retries=3)
        tool = _PermanentFailTool()
        registry.register(tool)

        result = await registry.dispatch(
            ToolCall(tool_name="permanent", arguments={}),
            ToolContext(cwd=Path("."), session_id="test"),
        )
        assert result.is_error
        assert "bad argument" in result.error

    @pytest.mark.asyncio
    async def test_backoff_durations_exponential(self) -> None:
        registry = ToolRegistry(max_retries=3)
        tool = _FlakyTool(fail_count=3)
        registry.register(tool)

        sleeps: list[float] = []

        async def capture_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            await registry.dispatch(
                ToolCall(tool_name="flaky", arguments={}),
                ToolContext(cwd=Path("."), session_id="test"),
            )

        # Exponential backoff: 0.1, 0.2, 0.4
        assert sleeps == [0.1, 0.2, 0.4]

    @pytest.mark.asyncio
    async def test_unknown_tool_no_retry(self) -> None:
        registry = ToolRegistry(max_retries=3)
        result = await registry.dispatch(
            ToolCall(tool_name="missing", arguments={}),
            ToolContext(cwd=Path("."), session_id="test"),
        )
        assert result.is_error
        assert "Unknown tool" in result.error
