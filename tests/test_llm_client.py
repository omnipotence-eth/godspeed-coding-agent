"""Tests for LLM client — ChatResponse, ModelRouter, LLMClient, BudgetExceededError."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.llm.client import (
    BudgetExceededError,
    ChatResponse,
    LLMClient,
    ModelRouter,
)

# ---------------------------------------------------------------------------
# Test: ChatResponse
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_has_tool_calls_true(self) -> None:
        r = ChatResponse(tool_calls=[{"id": "1", "function": {"name": "x"}}])
        assert r.has_tool_calls is True

    def test_has_tool_calls_false(self) -> None:
        r = ChatResponse(content="hello")
        assert r.has_tool_calls is False

    def test_defaults(self) -> None:
        r = ChatResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.thinking == ""


# ---------------------------------------------------------------------------
# Test: BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_attributes(self) -> None:
        err = BudgetExceededError(spent=1.50, limit=1.00)
        assert err.spent == 1.50
        assert err.limit == 1.00
        assert "$1.5" in str(err)


# ---------------------------------------------------------------------------
# Test: ModelRouter
# ---------------------------------------------------------------------------


class TestModelRouter:
    def test_no_routing(self) -> None:
        router = ModelRouter()
        assert router.route("ollama/qwen3:4b") == "ollama/qwen3:4b"
        assert router.has_routing is False

    def test_with_routing(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet", "edit": "gpt-4o"})
        assert router.route("default", "plan") == "claude-sonnet"
        assert router.route("default", "edit") == "gpt-4o"
        assert router.route("default", "chat") == "default"
        assert router.has_routing is True
        assert router.routes == {"plan": "claude-sonnet", "edit": "gpt-4o"}

    def test_unknown_task_type_uses_default(self) -> None:
        router = ModelRouter({"plan": "claude"})
        assert router.route("fallback", "unknown") == "fallback"

    def test_none_task_type(self) -> None:
        router = ModelRouter({"plan": "claude"})
        assert router.route("default", None) == "default"


# ---------------------------------------------------------------------------
# Test: LLMClient helpers (no LLM calls)
# ---------------------------------------------------------------------------


class TestLLMClientHelpers:
    def test_supports_tool_calling_ollama_qwen(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        assert client._supports_tool_calling() is True

    def test_supports_tool_calling_ollama_unknown(self) -> None:
        client = LLMClient(model="ollama/phi4:latest")
        assert client._supports_tool_calling() is False

    def test_supports_tool_calling_api_model(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._supports_tool_calling() is True

    def test_effective_model_upgrades_ollama(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        assert client._effective_model() == "ollama_chat/qwen3:4b"

    def test_effective_model_no_upgrade_unsupported(self) -> None:
        client = LLMClient(model="ollama/phi4:latest")
        assert client._effective_model() == "ollama/phi4:latest"

    def test_effective_model_api_unchanged(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._effective_model() == "claude-sonnet-4-20250514"

    def test_is_connection_error(self) -> None:
        assert LLMClient._is_connection_error(ConnectionError("Connection refused"))
        assert not LLMClient._is_connection_error(ValueError("bad input"))

    def test_is_anthropic_model(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._is_anthropic_model() is True
        assert client._is_anthropic_model("anthropic/claude-haiku") is True
        client2 = LLMClient(model="gpt-4o")
        assert client2._is_anthropic_model() is False

    def test_check_budget_no_limit(self) -> None:
        client = LLMClient(model="test", max_cost_usd=0.0)
        client.total_cost_usd = 100.0
        client._check_budget()  # Should not raise

    def test_check_budget_exceeded(self) -> None:
        client = LLMClient(model="test", max_cost_usd=1.0)
        client.total_cost_usd = 1.50
        with pytest.raises(BudgetExceededError):
            client._check_budget()

    def test_check_budget_within_limit(self) -> None:
        client = LLMClient(model="test", max_cost_usd=2.0)
        client.total_cost_usd = 1.50
        client._check_budget()  # Should not raise

    def test_build_failure_error_ollama(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        err = client._build_failure_error(ConnectionError("Connection refused"))
        assert "ollama serve" in str(err).lower()

    def test_build_failure_error_generic(self) -> None:
        client = LLMClient(model="gpt-4o")
        err = client._build_failure_error(RuntimeError("timeout"))
        assert "All models failed" in str(err)

    def test_apply_prompt_caching_claude(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Query"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet", messages)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_apply_prompt_caching_ollama_noop(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        result = LLMClient._apply_prompt_caching("ollama/qwen3", messages)
        assert result == messages


# ---------------------------------------------------------------------------
# Test: LLMClient.chat (mocked LiteLLM)
# ---------------------------------------------------------------------------


def _mock_response(
    content: str = "Hello",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    """Build a mock litellm response."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        thinking=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class TestLLMClientChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = _mock_response(content="Hi there")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "hello"}])

        assert response.content == "Hi there"
        assert response.finish_reason == "stop"
        assert client.total_input_tokens == 10
        assert client.total_output_tokens == 20

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        tc = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(name="file_read", arguments='{"path": "foo.py"}'),
        )
        mock_resp = _mock_response(content="", tool_calls=[tc])

        client = LLMClient(model="ollama/qwen3:4b")
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "read foo"}])

        assert response.has_tool_calls
        assert response.tool_calls[0]["function"]["name"] == "file_read"

    @pytest.mark.asyncio
    async def test_chat_with_thinking(self) -> None:
        msg = SimpleNamespace(
            content="Result",
            tool_calls=[],
            thinking="I need to think about this...",
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)

        client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=1000)
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "think"}])

        assert response.thinking == "I need to think about this..."
        # Verify thinking param was passed
        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["thinking"]["budget_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_chat_content_blocks(self) -> None:
        """Handle responses where content is a list of blocks."""
        msg = SimpleNamespace(
            content=[
                {"type": "text", "text": "Part 1. "},
                {"type": "text", "text": "Part 2."},
            ],
            tool_calls=[],
            thinking=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=10)
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)

        client = LLMClient(model="claude-sonnet-4-20250514")
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "Part 1. Part 2."

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        call_count = 0

        async def _failing_then_success(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Primary + retry both fail
                raise RuntimeError("model down")
            return _mock_response(content="from fallback")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=_failing_then_success)
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "from fallback"
        assert call_count == 3  # primary + retry + fallback

    @pytest.mark.asyncio
    async def test_all_models_fail_raises(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=RuntimeError("down"))
            with pytest.raises(RuntimeError, match="All models failed"):
                await client.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_budget_exceeded_during_chat(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514", max_cost_usd=0.001)
        mock_resp = _mock_response(input_tokens=100_000, output_tokens=50_000)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            # BudgetExceededError is caught by fallback loop, surfaced as RuntimeError
            with pytest.raises(RuntimeError, match="Budget exceeded"):
                await client.chat([{"role": "user", "content": "expensive"}])

    @pytest.mark.asyncio
    async def test_model_routing(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet-4-20250514"})
        client = LLMClient(model="ollama/qwen3:4b", router=router)
        mock_resp = _mock_response()

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            await client.chat(
                [{"role": "user", "content": "plan this"}],
                task_type="plan",
            )

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        # Model should be restored after call
        assert client.model == "ollama/qwen3:4b"

    @pytest.mark.asyncio
    async def test_connection_refused_skips_retry(self) -> None:
        """Connection-refused errors skip the retry and go straight to fallback."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        call_count = 0

        async def _conn_refused_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["model"].startswith("ollama_chat/qwen"):
                raise ConnectionError("Connection refused")
            return _mock_response(content="ok")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=_conn_refused_then_ok)
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "ok"
        # Should be: primary (fail, conn refused) + fallback (ok) = 2 calls
        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Retry / fallback / error-classification helpers
# ---------------------------------------------------------------------------


class TestRetryHelpers:
    """Unit tests for static/class retry methods on LLMClient."""

    def test_is_rate_limit_429_pair(self) -> None:
        assert LLMClient._is_rate_limit_error(RuntimeError("429 too many requests"))
        assert LLMClient._is_rate_limit_error(RuntimeError("429 rate limit hit"))
        assert LLMClient._is_rate_limit_error(RuntimeError("429 quota exceeded"))
        assert LLMClient._is_rate_limit_error(RuntimeError("429 throttled"))

    def test_is_rate_limit_marker(self) -> None:
        assert LLMClient._is_rate_limit_error(RuntimeError("rate_limit_exceeded"))
        assert LLMClient._is_rate_limit_error(RuntimeError("rate limit"))
        assert LLMClient._is_rate_limit_error(RuntimeError("ratelimiterror"))
        assert LLMClient._is_rate_limit_error(RuntimeError("quota exceeded"))
        assert LLMClient._is_rate_limit_error(RuntimeError("too many requests"))

    def test_is_rate_limit_bare_429_no_match(self) -> None:
        """A bare '429' without a rate-limit keyword nearby should not match."""
        assert not LLMClient._is_rate_limit_error(RuntimeError("port 429 is in use"))
        assert not LLMClient._is_rate_limit_error(RuntimeError("error code 429"))

    def test_is_rate_limit_no_match(self) -> None:
        assert not LLMClient._is_rate_limit_error(RuntimeError("internal server error"))
        assert not LLMClient._is_rate_limit_error(ValueError("bad input"))

    def test_parse_retry_after_none(self) -> None:
        assert LLMClient._parse_retry_after("rate limit hit") is None
        assert LLMClient._parse_retry_after("") is None

    def test_parse_retry_after_variants(self) -> None:
        assert LLMClient._parse_retry_after("retry-after: 5") == 5.0
        assert LLMClient._parse_retry_after("Retry-After: 30") == 30.0
        assert LLMClient._parse_retry_after("retryafter: 15") == 15.0
        assert LLMClient._parse_retry_after("RetryAfter: 20") == 20.0

    def test_parse_retry_after_clamps_to_max(self) -> None:
        from godspeed.llm.client import RATE_LIMIT_MAX_DELAY

        result = LLMClient._parse_retry_after(f"retry-after: {RATE_LIMIT_MAX_DELAY + 100}")
        assert result == RATE_LIMIT_MAX_DELAY

    def test_backoff_delay_with_retry_after(self) -> None:
        with patch("random.uniform", return_value=0.1):
            delay = LLMClient._backoff_delay(0, retry_after=5.0)
        # 5.0 * (1.0 + 0.1) = 5.5
        assert abs(delay - 5.5) < 0.01

    def test_backoff_delay_exponential(self) -> None:
        with patch("random.uniform", return_value=0.0):  # no jitter
            d0 = LLMClient._backoff_delay(0, retry_after=None)
            d1 = LLMClient._backoff_delay(1, retry_after=None)
            d2 = LLMClient._backoff_delay(2, retry_after=None)
        assert abs(d0 - 1.0) < 0.01  # base * 2^0 = 1
        assert abs(d1 - 2.0) < 0.01  # base * 2^1 = 2
        assert abs(d2 - 4.0) < 0.01  # base * 2^2 = 4

    def test_backoff_delay_caps_at_max(self) -> None:
        from godspeed.llm.client import RATE_LIMIT_MAX_DELAY

        with patch("random.uniform", return_value=1.25):  # maximum +jitter
            delay = LLMClient._backoff_delay(100, retry_after=None)
        assert delay <= RATE_LIMIT_MAX_DELAY


class TestLLMClientRateLimit:
    """Tests for _retry_on_rate_limit and rate-limit integration in chat()."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_recovers(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("429 rate limit hit"),  # first call fails
                    _mock_response(content="recovered"),  # retry succeeds
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client._retry_on_rate_limit(
                    "ollama_chat/qwen3:4b",
                    [{"role": "user", "content": "hi"}],
                    None,
                    RuntimeError("429 rate limit hit"),
                )

        assert response is not None
        assert response.content == "recovered"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_exhausted(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=RuntimeError("429 rate limit hit"),
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client._retry_on_rate_limit(
                    "ollama_chat/qwen3:4b",
                    [{"role": "user", "content": "hi"}],
                    None,
                    RuntimeError("429 rate limit hit"),
                )

        assert response is None

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_morphs_error(self) -> None:
        """Non-rate-limit error during retry should propagate immediately."""
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=RuntimeError("connection refused"),
            )
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(RuntimeError, match="connection refused"):
                    await client._retry_on_rate_limit(
                        "ollama_chat/qwen3:4b",
                        [{"role": "user", "content": "hi"}],
                        None,
                        RuntimeError("429 rate limit hit"),
                    )

    @pytest.mark.asyncio
    async def test_rate_limit_falls_to_fallback(self) -> None:
        """Rate-limit exhausts on primary, moves to fallback model."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        call_count = 0

        async def _rate_limit_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if "qwen" in kwargs["model"]:
                raise RuntimeError("429 rate limit hit")
            return _mock_response(content="from fallback")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=_rate_limit_then_ok)
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "from fallback"
        # primary + 4 rate-limit retries + fallback
        assert call_count >= 5

    @pytest.mark.asyncio
    async def test_chat_rate_limit_then_recovers_on_primary(self) -> None:
        """Rate-limit triggers retry, then recovers on same model."""
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("429 rate limit hit"),
                    _mock_response(content="ok after backoff"),
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "ok after backoff"

    @pytest.mark.asyncio
    async def test_generic_error_triggers_primary_retry(self) -> None:
        """Non-rate-limit, non-connection error retries primary once."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("server error"),  # primary fails
                    _mock_response(content="retry ok"),  # primary retry succeeds
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "retry ok"

    @pytest.mark.asyncio
    async def test_generic_error_primary_retry_then_fallback(self) -> None:
        """Primary retry fails, falls through to fallback model."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("server error"),  # primary
                    RuntimeError("server error again"),  # primary retry
                    _mock_response(content="fallback ok"),  # fallback
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "fallback ok"


class TestLLMClientCall:
    """Tests for the internal _call method."""

    @pytest.mark.asyncio
    async def test_call_basic(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=[], thinking=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "hi"}],
                None,
            )

        assert response.content == "hello"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_call_empty_choices(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = SimpleNamespace(choices=[], usage=None)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "hi"}],
                None,
            )

        assert response.content == ""
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_call_with_tools(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        tc = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="file_read", arguments='{"path": "x"}'),
        )
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[tc], thinking=None),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "read x"}],
                [{"type": "function", "function": {"name": "file_read"}}],
            )

        assert response.has_tool_calls
        assert response.tool_calls[0]["function"]["name"] == "file_read"


class TestLLMClientConnectionError:
    """Edge-case tests for connection error detection and messaging."""

    def test_is_connection_error_variants(self) -> None:
        assert LLMClient._is_connection_error(ConnectionError("Connection refused"))
        assert LLMClient._is_connection_error(OSError("Cannot connect"))
        assert LLMClient._is_connection_error(RuntimeError("Connect call failed"))
        assert not LLMClient._is_connection_error(RuntimeError("timeout"))
        assert not LLMClient._is_connection_error(ValueError("something else"))

    def test_build_failure_error_llamacpp(self) -> None:
        client = LLMClient(model="llamacpp/qwen2.5-coder")
        err = client._build_failure_error(ConnectionError("Connection refused"))
        assert "llama.cpp" in str(err)
        assert "not running" in str(err)

    def test_build_failure_error_llamacpp_openai(self) -> None:
        client = LLMClient(model="openai/qwen2.5-coder")
        err = client._build_failure_error(ConnectionError("Connection refused"))
        assert "llama.cpp" in str(err)

    @pytest.mark.asyncio
    async def test_connection_error_no_retry(self) -> None:
        """Connection errors should not trigger a retry — go straight to fallback."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    ConnectionError("Connection refused"),  # primary, no retry
                    _mock_response(content="fallback ok"),  # fallback
                ]
            )
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "fallback ok"
