"""LLM client wrapping LiteLLM for unified model access."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


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

    def _supports_tool_calling(self) -> bool:
        """Check if current model likely supports native function calling."""
        model_lower = self.model.lower()
        # Models known to NOT support tool calling
        no_tools = ("ollama/", "ollama_chat/")
        # Exception: some Ollama models DO support tools
        tools_capable_ollama = (
            "qwen",
            "llama3",
            "mistral",
            "command-r",
            "firefunction",
            "hermes",
        )

        if any(model_lower.startswith(prefix) for prefix in no_tools):
            model_name = model_lower.split("/", 1)[-1].split(":")[0]
            return any(cap in model_name for cap in tools_capable_ollama)
        return True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send messages to the LLM and return a parsed response.

        Uses LiteLLM's async completion with automatic provider routing.
        Falls back to alternate models on failure.
        """
        models_to_try = [self.model, *self.fallback_models]

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                return await self._call(model, messages, tools)
            except TimeoutError as exc:
                msg = f"LLM call timed out after {self.timeout}s for model={model}"
                logger.warning(msg)
                last_error = TimeoutError(msg)
                # Retry primary model once after short delay before trying fallbacks
                if idx == 0:
                    logger.warning("Primary model failed, retrying once after 2s delay: %s", exc)
                    await asyncio.sleep(2)
                    try:
                        return await self._call(model, messages, tools)
                    except Exception as retry_exc:
                        logger.warning(
                            "Primary model retry failed model=%s error=%s",
                            model,
                            retry_exc,
                        )
                        last_error = retry_exc
            except Exception as exc:
                logger.warning("LLM call failed model=%s error=%s", model, exc)
                # Retry primary model once after short delay before trying fallbacks
                if idx == 0:
                    logger.warning("Primary model failed, retrying once after 2s delay: %s", exc)
                    await asyncio.sleep(2)
                    try:
                        return await self._call(model, messages, tools)
                    except Exception as retry_exc:
                        logger.warning(
                            "Primary model retry failed model=%s error=%s",
                            model,
                            retry_exc,
                        )
                        last_error = retry_exc
                else:
                    last_error = exc

        msg = f"All models failed. Last error: {last_error}"
        raise RuntimeError(msg)

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

        response = await litellm.acompletion(**kwargs)

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
        kwargs: dict[str, Any] = {
            "model": self.model,
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
                    self.model,
                )

        try:
            response = await litellm.acompletion(**kwargs)
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
