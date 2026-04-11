"""LLM client wrapping LiteLLM for unified model access."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class ChatResponse:
    """Parsed response from an LLM call."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


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
    ) -> None:
        self.model = model
        self.fallback_models = fallback_models or []
        self.timeout = timeout
        self.total_input_tokens = 0
        self.total_output_tokens = 0

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

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send messages to the LLM and return a parsed response.

        Uses LiteLLM's async completion with automatic provider routing.
        Falls back to alternate models on failure. Skips retries for
        connection-refused errors (e.g. Ollama not running).
        """
        models_to_try = [self._effective_model(), *self.fallback_models]

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                return await self._call(model, messages, tools)
            except Exception as exc:
                logger.warning("LLM call failed model=%s error=%s", model, exc)
                last_error = exc
                # Skip retry for connection errors — server is down, retrying is pointless
                if self._is_connection_error(exc):
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

    async def _call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        """Make a single LLM API call."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
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

        response = await _get_litellm().acompletion(**kwargs)

        # Parse response
        choice = response.choices[0]
        message = choice.message

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

        # Track usage
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0

        return ChatResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage={
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }
            if response.usage
            else {},
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Stream LLM response chunks.

        Yields ChatResponse objects as they arrive. The final response
        has finish_reason set.
        """
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

        try:
            response = await _get_litellm().acompletion(**kwargs)
            collected_content = ""
            collected_tool_calls: list[dict[str, Any]] = []

            async for chunk in response:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

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
