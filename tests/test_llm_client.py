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
        messages = [{"role": "system", "content": "You are helpful."}]
        result = LLMClient._apply_prompt_caching("claude-sonnet", messages)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_apply_prompt_caching_ollama_noop(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        result = LLMClient._apply_prompt_caching("ollama/qwen3", messages)
        assert result == messages

    def test_apply_prompt_caching_marks_last_stable_turn(self) -> None:
        # System + user + assistant + tool + user — breakpoint should
        # land on the ``tool`` (last stable turn before the final user
        # message), in addition to the system prompt.
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "ls"}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "file1.py file2.py"},
            {"role": "user", "content": "now read file1.py"},
        ]
        result = LLMClient._apply_prompt_caching("claude-opus-4-7", messages)

        # System prompt remains cached.
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Last user message untouched (string content) — that's the new input.
        assert result[4]["content"] == "now read file1.py"
        assert not isinstance(result[4]["content"], list)
        # Tool result (idx 3) is wrapped with cache_control.
        assert isinstance(result[3]["content"], list)
        assert result[3]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_apply_prompt_caching_denylist_skipped(self) -> None:
        # Gemini has a different caching API; we should never emit
        # cache_control for it.
        messages = [{"role": "system", "content": "helpful"}]
        result = LLMClient._apply_prompt_caching("gemini/gemini-2.0-flash", messages)
        assert result == messages

    def test_apply_prompt_caching_bedrock_claude(self) -> None:
        # Anthropic models served via Bedrock share the same API shape.
        messages = [{"role": "system", "content": "helpful"}]
        result = LLMClient._apply_prompt_caching(
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", messages
        )
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_prompt_caching_flag_off_bypasses_apply(self) -> None:
        # When the flag is off we expect _call to short-circuit the
        # caching wrapper — the messages LiteLLM sees match what we passed.
        from unittest.mock import AsyncMock, MagicMock, patch

        client = LLMClient(model="claude-sonnet-4", prompt_caching=False)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message = MagicMock(content="ok", tool_calls=None)
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            messages = [{"role": "system", "content": "You are helpful."}]
            import asyncio

            asyncio.run(client._call("claude-sonnet-4", messages, None))

        # Inspect what litellm was called with — messages should be
        # the exact plain-string form, no cache_control injection.
        sent_kwargs = mock_litellm.return_value.acompletion.call_args.kwargs
        assert sent_kwargs["messages"] == messages

    def test_cache_hit_telemetry_recorded(self) -> None:
        # LiteLLM surfaces cache_read_input_tokens on usage when
        # Anthropic reports one — the client must accumulate it so
        # /stats can report the savings.
        from unittest.mock import AsyncMock, MagicMock, patch

        client = LLMClient(model="claude-sonnet-4")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message = MagicMock(content="ok", tool_calls=None)
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(
            prompt_tokens=1000,
            completion_tokens=50,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=0,
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            import asyncio

            asyncio.run(
                client._call(
                    "claude-sonnet-4",
                    [{"role": "system", "content": "sys"}],
                    None,
                )
            )
        assert client.total_cache_read_tokens == 800
        assert client.total_cache_creation_tokens == 0


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
