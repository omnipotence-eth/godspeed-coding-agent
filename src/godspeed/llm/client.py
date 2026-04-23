"""LLM client wrapping LiteLLM for unified model access."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

# Rate-limit retry policy
RATE_LIMIT_MAX_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 1.0  # seconds — doubles each retry
RATE_LIMIT_MAX_DELAY = 60.0  # hard ceiling — past this, give up and fall over
RATE_LIMIT_JITTER = 0.25  # ±25% random jitter on each delay

_RATE_LIMIT_MARKERS = (
    "429",
    "rate_limit",
    "rate limit",
    "ratelimiterror",
    "too many requests",
    "quota",
)
_RETRY_AFTER_RE = re.compile(r"retry-?after[:\s]+(\d+)", re.IGNORECASE)

logger = logging.getLogger(__name__)

# Lazy import — litellm pulls in 2000+ modules (~1.5s cold start).
# We defer it to first use so the TUI appears instantly.
_litellm = None


def _get_litellm():
    """Import litellm on first use and cache it."""
    global _litellm
    if _litellm is None:
        import litellm

        litellm.suppress_debug_info = True
        _litellm = litellm
    return _litellm


class BudgetExceededError(RuntimeError):
    """Raised when session cost exceeds the configured budget."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: ${spent:.4f} / ${limit:.2f} limit")


@dataclass
class ChatResponse:
    """Parsed response from an LLM call."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = ""
    usage: dict[str, int] = field(default_factory=dict)
    thinking: str = ""  # Extended thinking content (Anthropic models)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


def _extract_thinking_delta(delta: Any) -> str:
    """Pull a streaming extended-thinking chunk out of a LiteLLM delta.

    LiteLLM surfaces Anthropic extended-thinking deltas two different ways
    depending on provider/version:

    - ``delta.thinking_blocks`` — a list of ``{"type": "thinking", "thinking": "..."}``
      dicts (newer; matches the final-message content-block shape).
    - ``delta.thinking`` — a plain string (older / some proxies).

    Return the concatenated text for this chunk, or ``""`` if neither
    field is present. Never raises; unknown shapes are ignored so the
    stream continues.
    """
    text = ""
    blocks = getattr(delta, "thinking_blocks", None)
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict):
                piece = block.get("thinking") or block.get("text") or ""
                if isinstance(piece, str):
                    text += piece
    plain = getattr(delta, "thinking", None)
    if isinstance(plain, str):
        text += plain
    return text


class ModelRouter:
    """Routes LLM calls to different models based on task type.

    Config maps task types to model names. Unmatched task types
    fall back to the default model.
    """

    def __init__(self, routing: dict[str, str] | None = None) -> None:
        self._routing = routing or {}

    def route(self, default_model: str, task_type: str | None = None) -> str:
        """Select the model for a given task type.

        Args:
            default_model: The default model to use.
            task_type: Optional task hint (e.g., "plan", "edit", "chat").

        Returns:
            The model to use for this call.
        """
        if task_type and task_type in self._routing:
            routed = self._routing[task_type]
            logger.debug("Model routing task_type=%s model=%s", task_type, routed)
            return routed
        return default_model

    @property
    def has_routing(self) -> bool:
        return bool(self._routing)

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routing)


class LLMClient:
    """Unified LLM client via LiteLLM.

    Supports 200+ providers: Claude, GPT, Gemini, DeepSeek, Ollama, etc.
    Provides fallback chains, streaming, and token tracking.
    """

    def __init__(
        self,
        model: str,
        fallback_models: list[str] | None = None,
        timeout: int = 120,
        router: ModelRouter | None = None,
        thinking_budget: int = 0,
        max_cost_usd: float = 0.0,
        prompt_caching: bool = True,
    ) -> None:
        self.model = model
        self.fallback_models = fallback_models or []
        self.timeout = timeout
        self.router = router or ModelRouter()
        self.thinking_budget = thinking_budget
        self.max_cost_usd = max_cost_usd
        self.prompt_caching = prompt_caching
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd: float = 0.0
        # Cache-hit telemetry — populated per-call from the provider's
        # cache_read_input_tokens / cache_creation_input_tokens fields
        # (Anthropic) or its equivalent. 0 means "unknown / not
        # applicable" rather than "zero hits".
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0

    # Ollama models known to support native tool calling
    _TOOLS_CAPABLE_OLLAMA = (
        "qwen",
        "llama3",
        "mistral",
        "command-r",
        "firefunction",
        "hermes",
        "gemma",
    )

    def _supports_tool_calling(self) -> bool:
        """Check if current model likely supports native function calling."""
        model_lower = self.model.lower()
        if model_lower.startswith(("ollama/", "ollama_chat/")):
            model_name = model_lower.split("/", 1)[-1].split(":")[0]
            return any(cap in model_name for cap in self._TOOLS_CAPABLE_OLLAMA)
        return True

    def _effective_model(self) -> str:
        """Return the model string to use for API calls.

        Upgrades 'ollama/' to 'ollama_chat/' for tool-capable models,
        since LiteLLM's ollama_chat provider supports native tool calling
        while the plain ollama provider does not.
        """
        model_lower = self.model.lower()
        if model_lower.startswith("ollama/") and self._supports_tool_calling():
            upgraded = "ollama_chat/" + self.model.split("/", 1)[1]
            logger.info("Upgrading model %s → %s for tool calling support", self.model, upgraded)
            return upgraded
        return self.model

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Check if the error is a connection failure (server not running)."""
        exc_str = str(exc).lower()
        return any(
            marker in exc_str
            for marker in ("connection refused", "cannot connect", "connect call failed")
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if the error is a 429 / rate limit / quota error.

        These are transient and should be retried with exponential backoff,
        not failed over immediately (falling over from rate-limited primary
        to a fallback often just rate-limits the fallback too).
        """
        exc_str = str(exc).lower()
        # "429" appears in many unrelated contexts (ports, request IDs) — only
        # count it when paired with a rate-limit word nearby.
        if "429" in exc_str and any(w in exc_str for w in ("too many", "rate", "quota", "throttl")):
            return True
        return any(marker in exc_str for marker in _RATE_LIMIT_MARKERS if marker != "429")

    @staticmethod
    def _parse_retry_after(error_message: str) -> float | None:
        """Extract a Retry-After hint (seconds) from the error message.

        Returns None when no hint is present. Clamps to RATE_LIMIT_MAX_DELAY
        so a misbehaving provider can't block the session for hours.
        """
        match = _RETRY_AFTER_RE.search(error_message)
        if match is None:
            return None
        try:
            hint = float(match.group(1))
        except ValueError:
            return None
        return min(hint, RATE_LIMIT_MAX_DELAY)

    @classmethod
    def _backoff_delay(cls, retry_index: int, retry_after: float | None) -> float:
        """Compute the sleep duration for the N-th retry (0-indexed).

        If the provider supplied Retry-After, treat it as a floor and add
        upward-only jitter (waiting *less* than the provider asked is
        counterproductive — we'd just trigger another 429).

        Otherwise use exponential backoff (base * 2^n) with ±25% jitter
        to break up thundering-herd retries across concurrent agents.
        Capped at RATE_LIMIT_MAX_DELAY.
        """
        if retry_after is not None:
            # Retry jitter for backoff — not a security context, so random.uniform is fine.
            jitter = 1.0 + random.uniform(0.0, RATE_LIMIT_JITTER)  # noqa: S311
            return min(retry_after * jitter, RATE_LIMIT_MAX_DELAY)
        base = min(RATE_LIMIT_BASE_DELAY * (2**retry_index), RATE_LIMIT_MAX_DELAY)
        jitter = 1.0 + random.uniform(-RATE_LIMIT_JITTER, RATE_LIMIT_JITTER)  # noqa: S311
        return min(max(base * jitter, 0.0), RATE_LIMIT_MAX_DELAY)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_type: str | None = None,
    ) -> ChatResponse:
        """Send messages to the LLM and return a parsed response.

        Uses LiteLLM's async completion with automatic provider routing.
        Falls back to alternate models on failure. Skips retries for
        connection-refused errors (e.g. Ollama not running).

        Args:
            messages: Conversation messages.
            tools: Tool schemas for function calling.
            task_type: Optional task hint for model routing
                (e.g., "plan", "edit", "chat").
        """
        # Apply model routing based on task type
        routed_model = self.router.route(self.model, task_type)
        if routed_model != self.model:
            # Temporarily use the routed model
            original_model = self.model
            self.model = routed_model
            try:
                return await self._chat_with_fallback(messages, tools)
            finally:
                self.model = original_model
        return await self._chat_with_fallback(messages, tools)

    async def _chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Internal: send messages with fallback chain.

        Classification of failures:
        - Connection errors: server is down; skip retry, try next fallback.
        - Rate-limit / 429 / quota: retry the SAME model with exponential
          backoff + jitter up to RATE_LIMIT_MAX_RETRIES, honoring
          Retry-After when provided. Falling over to a fallback on
          rate-limit often just rate-limits the fallback too.
        - Other errors: one short retry on the primary model, then fall
          over to the next model in the chain.
        """
        models_to_try = [self._effective_model(), *self.fallback_models]

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                return await self._call(model, messages, tools)
            except Exception as exc:
                logger.warning("LLM call failed model=%s error=%s", model, exc)
                last_error = exc

                if self._is_connection_error(exc):
                    # Server down — retrying is pointless, try next fallback.
                    continue

                if self._is_rate_limit_error(exc):
                    # Retry same model with exponential backoff + jitter.
                    recovered = await self._retry_on_rate_limit(model, messages, tools, exc)
                    if recovered is not None:
                        return recovered
                    # Exhausted rate-limit retries; move on to the next model.
                    last_error = exc
                    continue

                # Retry primary model once after short delay before trying fallbacks
                if idx == 0:
                    await asyncio.sleep(1)
                    try:
                        return await self._call(model, messages, tools)
                    except Exception as retry_exc:
                        logger.warning(
                            "Primary model retry failed model=%s error=%s",
                            model,
                            retry_exc,
                        )
                        last_error = retry_exc

        raise self._build_failure_error(last_error)

    async def _retry_on_rate_limit(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        first_exc: Exception,
    ) -> ChatResponse | None:
        """Retry a rate-limited call with exponential backoff + jitter.

        Returns the successful ChatResponse, or None if retries are exhausted.
        Caller decides what to do with a None result (typically fall over to
        the next model in the chain).
        """
        current_exc: Exception = first_exc
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            retry_after = self._parse_retry_after(str(current_exc))
            delay = self._backoff_delay(attempt, retry_after)
            logger.warning(
                "Rate limit model=%s attempt=%d/%d delay=%.2fs retry_after=%s",
                model,
                attempt + 1,
                RATE_LIMIT_MAX_RETRIES,
                delay,
                retry_after,
            )
            await asyncio.sleep(delay)
            try:
                return await self._call(model, messages, tools)
            except Exception as exc:
                if not self._is_rate_limit_error(exc):
                    # Morphed into a different kind of failure — let the
                    # outer loop handle it (fall over, build failure, etc.).
                    raise
                current_exc = exc
        logger.warning(
            "Rate limit retries exhausted model=%s after %d attempts",
            model,
            RATE_LIMIT_MAX_RETRIES,
        )
        return None

    def _build_failure_error(self, last_error: Exception | None) -> RuntimeError:
        """Build an actionable error message based on the failure type."""
        if last_error and self._is_connection_error(last_error):
            model_lower = self.model.lower()
            if model_lower.startswith("ollama"):
                return RuntimeError(
                    "Ollama is not running. Fix with one of:\n"
                    "  1. Start Ollama:  ollama serve\n"
                    "  2. Use a cloud model:  godspeed -m claude-sonnet-4-20250514\n"
                    "  3. Set a fallback in ~/.godspeed/settings.yaml"
                )
            return RuntimeError(
                f"Cannot connect to LLM provider for model '{self.model}'. "
                "Check that the server is running and the model name is correct."
            )
        return RuntimeError(f"All models failed. Last error: {last_error}")

    # Provider substrings that accept explicit ``cache_control`` markers
    # (Anthropic's API shape — directly, or via Bedrock / Vertex proxies).
    # OpenAI and DeepSeek also accept them but apply caching automatically
    # based on prefix hashing, so marking costs nothing and helps nothing
    # there. Including the common OpenAI prefixes here keeps behavior
    # harmlessly compatible with litellm's pass-through for models that
    # emit the marker back as an error; any provider that errors on the
    # marker can be added to ``_CACHING_DENYLIST``.
    _CACHING_ALLOWLIST: tuple[str, ...] = (
        "claude",
        "anthropic",
        "bedrock/anthropic",
        "vertex_ai/claude",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4.1",
        "o1",
        "o3",
        "o4",
        "deepseek",
    )
    # Providers that reject or ignore ``cache_control`` noisily — skip.
    _CACHING_DENYLIST: tuple[str, ...] = (
        "ollama",
        "groq",
        "gemini",  # uses separate context-caching API shape
        "mistral",
    )

    @classmethod
    def _supports_cache_control(cls, model: str) -> bool:
        """Return True when the model accepts ``cache_control`` markers."""
        model_lower = model.lower()
        if any(bad in model_lower for bad in cls._CACHING_DENYLIST):
            return False
        return any(prefix in model_lower for prefix in cls._CACHING_ALLOWLIST)

    @classmethod
    def _apply_prompt_caching(
        cls,
        model: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply prompt caching markers for supported providers.

        Strategy: two breakpoints for Anthropic-family models:

        1. End of the system prompt — caches the tool descriptions,
           project context, and any permanent instructions. First cache
           hit on call #2.
        2. End of the *last stable* conversation turn (the last
           ``tool``-role message, or the last ``assistant`` message when
           no tools were called). This caches the entire conversation
           history so only the newest user input + new assistant
           response are re-billed each turn.

        For cache-free providers (Ollama, Groq, Gemini, Mistral) this
        is a pure no-op. For OpenAI and DeepSeek the marker is a no-op
        at the provider side (they cache automatically by prefix hash)
        but harmless to include.

        Anthropic allows up to 4 breakpoints; we use 2 — adding more
        rarely increases hit rate enough to justify the complexity.
        """
        if not cls._supports_cache_control(model):
            return messages

        cached: list[dict[str, Any]] = []

        # Find the index of the last conversation message we want to
        # mark as a cache breakpoint. Working backwards so a model
        # receives cache_control on the *latest* stable position — any
        # new user input after this point will not invalidate the
        # cached prefix.
        last_cacheable_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].get("role")
            # Don't mark the final user message — the caller's newest
            # input — as cached. Cache the prefix before it.
            if role == "user" and i == len(messages) - 1:
                continue
            if role in {"tool", "assistant"}:
                last_cacheable_idx = i
                break

        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            # Breakpoint #1: system prompt.
            if role == "system" and isinstance(content, str):
                cached.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
                continue

            # Breakpoint #2: last stable assistant / tool-result turn.
            if idx == last_cacheable_idx and isinstance(content, str):
                cached.append(
                    {
                        **msg,
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
                continue

            cached.append(msg)

        return cached

    def _is_anthropic_model(self, model: str | None = None) -> bool:
        """Check if the model is an Anthropic/Claude model."""
        name = (model or self.model).lower()
        return any(prefix in name for prefix in ("claude", "anthropic"))

    def _check_budget(self) -> None:
        """Raise BudgetExceededError if session cost exceeds the limit."""
        if self.max_cost_usd > 0 and self.total_cost_usd > self.max_cost_usd:
            raise BudgetExceededError(self.total_cost_usd, self.max_cost_usd)

    async def _call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        """Make a single LLM API call."""
        from godspeed.llm.cost import estimate_cost

        # Apply prompt caching for supported providers (opt-out via
        # self.prompt_caching = False).
        cached_messages = (
            self._apply_prompt_caching(model, messages) if self.prompt_caching else messages
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": cached_messages,
            "timeout": self.timeout,
        }
        if tools:
            if self._supports_tool_calling():
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            else:
                logger.info(
                    "Model %s may not support native tool calling; using text mode",
                    model,
                )

        # Extended thinking for Anthropic models
        if self.thinking_budget > 0 and self._is_anthropic_model(model):
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}

        response = await _get_litellm().acompletion(**kwargs)

        # Parse response
        choice = response.choices[0]
        message = choice.message

        # Extract thinking content from Anthropic responses
        thinking_text = ""
        if hasattr(message, "thinking") and message.thinking:
            thinking_text = message.thinking
        # Also check content blocks for thinking type
        if not thinking_text and hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking", "")
                    break

        # Extract text content (may be in content blocks)
        content_text = ""
        if isinstance(message.content, str):
            content_text = message.content
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content_text += block.get("text", "")

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        # Qwen3-Coder-family models emit <function=...>...</function> XML that
        # Ollama's built-in parser (as of 0.20.x) doesn't extract — the call
        # ends up in the content field instead of tool_calls. When we see the
        # fingerprint and no structured tool_calls, parse and synthesize.
        if not tool_calls and content_text:
            from godspeed.llm.qwen3_coder_parser import extract_qwen3_coder_tool_calls

            parsed = extract_qwen3_coder_tool_calls(content_text)
            if parsed:
                tool_calls = parsed
                content_text = ""

        # Track usage and cost
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            # Cache hit telemetry — Anthropic returns these in the
            # usage block; LiteLLM surfaces them as-is when present.
            # Silent when the provider doesn't populate them.
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            if cache_read or cache_create:
                self.total_cache_read_tokens += cache_read
                self.total_cache_creation_tokens += cache_create
                logger.debug(
                    "Cache tokens model=%s read=%d created=%d",
                    model,
                    cache_read,
                    cache_create,
                )

        call_cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_cost_usd += call_cost

        # Check budget after tracking
        self._check_budget()

        return ChatResponse(
            content=content_text or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            thinking=thinking_text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if response.usage
            else {},
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_type: str | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Stream LLM response chunks.

        Yields ChatResponse objects as they arrive. The final response
        has finish_reason set.

        Args:
            messages: Conversation messages.
            tools: Tool schemas for function calling.
            task_type: Optional task hint for model routing (see
                :mod:`godspeed.llm.router`). When set and the router
                has a mapping, ``self.model`` is swapped for the
                duration of the stream and restored on cleanup.
        """
        # Apply task-aware routing for the duration of the stream.
        # The finally restores self.model even on early generator
        # close (aclose() in the agent loop's cancel path).
        routed_model = self.router.route(self.model, task_type)
        swap_model = routed_model != self.model
        original_model = self.model
        if swap_model:
            self.model = routed_model
        try:
            async for chunk in self._stream_chat_inner(messages, tools):
                yield chunk
        finally:
            if swap_model:
                self.model = original_model

    async def _stream_chat_inner(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Inner streaming body — assumes ``self.model`` is already routed."""
        effective = self._effective_model()
        kwargs: dict[str, Any] = {
            "model": effective,
            "messages": messages,
            "stream": True,
            "timeout": self.timeout,
        }
        if tools:
            if self._supports_tool_calling():
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            else:
                logger.info(
                    "Model %s may not support native tool calling; using text mode",
                    effective,
                )

        # Extended thinking for Anthropic models — parity with the non-
        # streaming path. Thinking blocks arrive interleaved with content
        # and are surfaced per-chunk below.
        if self.thinking_budget > 0 and self._is_anthropic_model(effective):
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}

        try:
            response = await _get_litellm().acompletion(**kwargs)
            collected_content = ""
            collected_thinking = ""
            collected_tool_calls: list[dict[str, Any]] = []

            async for chunk in response:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Extended thinking deltas (Anthropic) — LiteLLM surfaces
                # them on the delta as either ``delta.thinking_blocks`` (list)
                # or as a plain ``delta.thinking`` string, depending on
                # provider/version. Yield as a ChatResponse with only the
                # ``thinking`` field set; the agent loop routes it to the
                # TUI's ``on_thinking`` callback.
                think_text = _extract_thinking_delta(delta)
                if think_text:
                    collected_thinking += think_text
                    yield ChatResponse(
                        content="",
                        thinking=think_text,
                        tool_calls=[],
                        finish_reason=None,
                        usage={},
                    )

                if delta.content:
                    collected_content += delta.content
                    yield ChatResponse(
                        content=delta.content,
                        tool_calls=[],
                        finish_reason=None,
                        usage={},
                    )

                # Collect tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(collected_tool_calls) <= idx:
                            collected_tool_calls.append(
                                {"id": "", "function": {"name": "", "arguments": ""}}
                            )
                        if tc_delta.id:
                            collected_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                collected_tool_calls[idx]["function"]["name"] = (
                                    tc_delta.function.name
                                )
                            if tc_delta.function.arguments:
                                collected_tool_calls[idx]["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

                if finish_reason:
                    # Final chunk — yield complete response
                    final_tool_calls = []
                    for tc in collected_tool_calls:
                        final_tool_calls.append(
                            {
                                "id": tc["id"],
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                        )

                    chunk_usage = {}
                    if hasattr(chunk, "usage") and chunk.usage:
                        chunk_usage = dict(chunk.usage)

                    yield ChatResponse(
                        content=collected_content,
                        tool_calls=final_tool_calls,
                        finish_reason=finish_reason,
                        usage=chunk_usage,
                    )
                    return

            # Stream ended without finish_reason — return collected content
            if collected_content or collected_tool_calls:
                yield ChatResponse(
                    content=collected_content,
                    tool_calls=[],
                    finish_reason="incomplete",
                    usage={},
                )

        except Exception as exc:
            logger.error("Streaming LLM call failed: %s", exc, exc_info=True)
            raise
