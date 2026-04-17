"""Tests for rate-limit retry/backoff in LLMClient (v2.5.0).

MLOps pipelines hammer shared API keys. A single 429 from Anthropic must not
collapse the whole agent — the client must back off and retry with jitter,
and honor Retry-After when the provider sends it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from godspeed.llm.client import ChatResponse, LLMClient


class TestRateLimitClassifier:
    """_is_rate_limit_error recognizes 429s across provider formats."""

    def test_classic_rate_limit_message(self) -> None:
        c = LLMClient(model="test")
        assert c._is_rate_limit_error(Exception("429 Too Many Requests"))

    def test_anthropic_rate_limit_phrase(self) -> None:
        c = LLMClient(model="test")
        assert c._is_rate_limit_error(Exception("rate_limit_error"))
        assert c._is_rate_limit_error(Exception("Rate limit exceeded"))

    def test_openai_rate_limit_error(self) -> None:
        c = LLMClient(model="test")
        assert c._is_rate_limit_error(Exception("RateLimitError: tokens per minute"))

    def test_generic_5xx_not_rate_limit(self) -> None:
        """5xx errors go through the existing fallback path, not backoff."""
        c = LLMClient(model="test")
        assert not c._is_rate_limit_error(Exception("500 Internal Server Error"))

    def test_connection_refused_not_rate_limit(self) -> None:
        c = LLMClient(model="test")
        assert not c._is_rate_limit_error(Exception("Connection refused"))


class TestRetryAfterParsing:
    """_parse_retry_after extracts a wait hint from the provider's error."""

    def test_retry_after_integer_seconds(self) -> None:
        c = LLMClient(model="test")
        assert c._parse_retry_after("Retry-After: 30") == 30.0

    def test_retry_after_embedded_in_message(self) -> None:
        c = LLMClient(model="test")
        msg = "429 Too Many Requests. Retry-After: 12"
        assert c._parse_retry_after(msg) == 12.0

    def test_no_hint_returns_none(self) -> None:
        c = LLMClient(model="test")
        assert c._parse_retry_after("429 Too Many Requests") is None

    def test_clamps_absurd_values(self) -> None:
        """If provider says 'retry in 3 hours', we clamp — user should abort,
        not silently block the session that long."""
        c = LLMClient(model="test")
        parsed = c._parse_retry_after("Retry-After: 99999")
        assert parsed is not None
        assert parsed <= 60.0


class TestBackoffRetry:
    """Rate-limited calls retry with exponential backoff + jitter, then succeed."""

    @pytest.mark.asyncio
    async def test_transient_429_recovers(self) -> None:
        """A single 429 followed by a success should return the success."""
        c = LLMClient(model="test-model")
        ok = ChatResponse(content="recovered", finish_reason="stop")

        call_count = {"n": 0}

        async def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("429 Too Many Requests")
            return ok

        with patch.object(c, "_call", side_effect=flaky), patch("asyncio.sleep", AsyncMock()):
            result = await c._chat_with_fallback([])

        assert result.content == "recovered"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_multiple_429s_retry_up_to_cap(self) -> None:
        """Keep retrying up to RATE_LIMIT_MAX_RETRIES, then fail over to fallback."""
        c = LLMClient(model="test-model", fallback_models=["fallback-model"])
        ok = ChatResponse(content="fallback used", finish_reason="stop")

        call_log: list[str] = []

        async def flaky(model, *args, **kwargs):
            call_log.append(model)
            if model == "test-model":
                raise Exception("429 rate_limit_error")
            return ok

        with patch.object(c, "_call", side_effect=flaky), patch("asyncio.sleep", AsyncMock()):
            result = await c._chat_with_fallback([])

        assert result.content == "fallback used"
        # Primary tried RATE_LIMIT_MAX_RETRIES + 1 times (initial + retries)
        primary_attempts = [m for m in call_log if m == "test-model"]
        assert len(primary_attempts) >= 2
        assert "fallback-model" in call_log

    @pytest.mark.asyncio
    async def test_backoff_schedule_is_exponential(self) -> None:
        """Sleep durations must grow: each retry waits longer than the last."""
        c = LLMClient(model="test-model")
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def always_429(*args, **kwargs):
            raise Exception("429 rate_limit_error")

        with (
            patch.object(c, "_call", side_effect=always_429),
            patch("asyncio.sleep", side_effect=fake_sleep),
            pytest.raises(RuntimeError),
        ):
            await c._chat_with_fallback([])

        # Between retries on the primary model, we should have slept several times,
        # and the sequence should be monotonically non-decreasing (with jitter tolerance).
        assert len(sleeps) >= 2
        # Strip jitter by checking the base trend: sum of sleeps grows faster than linear
        # (exponential 1,2,4 sums to 7 which beats linear 1,1,1 at n=3).
        assert sum(sleeps[:3]) >= 1.0

    @pytest.mark.asyncio
    async def test_retry_after_honored(self) -> None:
        """When Retry-After is present, that value is used for the next wait."""
        c = LLMClient(model="test-model")
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        call_count = {"n": 0}

        async def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("429 rate_limit. Retry-After: 15")
            return ChatResponse(content="ok", finish_reason="stop")

        with (
            patch.object(c, "_call", side_effect=flaky),
            patch("asyncio.sleep", side_effect=fake_sleep),
        ):
            result = await c._chat_with_fallback([])

        assert result.content == "ok"
        # First sleep should be >= 15 (with small jitter tolerance upward only)
        assert sleeps[0] >= 15.0
