"""Conversation history management and compaction."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.llm.token_counter import count_message_tokens

logger = logging.getLogger(__name__)


class Conversation:
    """Manages the message history for an agent session.

    Tracks messages, monitors token usage, and triggers compaction
    when approaching the context limit.

    Optionally logs every message to a :class:`ConversationLogger` for
    fine-tuning data collection.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4",
        max_tokens: int = 100_000,
        compaction_threshold: float = 0.8,
        conversation_logger: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self._system_message: dict[str, Any] = {"role": "system", "content": system_prompt}
        self._messages: list[dict[str, Any]] = []
        self._logger = conversation_logger
        if self._logger is not None:
            self._logger.log_system(system_prompt)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return full message list including system prompt."""
        return [self._system_message, *self._messages]

    @property
    def token_count(self) -> int:
        """Estimate current token usage."""
        return count_message_tokens(self.messages, self.model)

    @property
    def is_near_limit(self) -> bool:
        """Check if we're approaching the context limit."""
        return self.token_count >= int(self.max_tokens * self.compaction_threshold)

    def add_user_message(self, content: str | list[dict[str, Any]]) -> None:
        """Add a user message.

        Args:
            content: Either a plain text string or a list of content blocks
                (e.g. text + image blocks in OpenAI multimodal format).
        """
        self._messages.append({"role": "user", "content": content})
        if self._logger is not None:
            self._logger.log_user(content)

    def add_assistant_message(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add an assistant message, optionally with tool calls."""
        msg: dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            # Ensure each tool call has 'type' — required by some providers
            # (e.g. LiteLLM's ollama_chat transformation)
            normalized = []
            for tc in tool_calls:
                entry = dict(tc)
                entry.setdefault("type", "function")
                normalized.append(entry)
            msg["tool_calls"] = normalized
        self._messages.append(msg)
        if self._logger is not None:
            self._logger.log_assistant(
                content=content,
                tool_calls=msg.get("tool_calls"),
            )

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message."""
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
        if self._logger is not None:
            self._logger.log_tool_result(
                tool_call_id=tool_call_id,
                tool_name="",  # caller can enrich via direct logger access
                content=content,
            )

    def compact(self, summary: str) -> None:
        """Replace conversation history with a summary.

        Preserves the system prompt and adds the summary as context,
        then clears the message history.
        """
        msg_count_before = len(self._messages)
        logger.info(
            "Compacting conversation tokens_before=%d message_count=%d",
            self.token_count,
            msg_count_before,
        )
        if self._logger is not None:
            self._logger.log_compaction(
                summary=summary,
                messages_before=msg_count_before,
                messages_after=1,
            )
        self._messages = [
            {
                "role": "user",
                "content": (
                    "[Conversation compacted. Summary of previous work:]\n\n"
                    f"{summary}\n\n"
                    "[Continue from where we left off.]"
                ),
            }
        ]
        logger.info("Compacted to tokens=%d", self.token_count)

    def get_compaction_context(self) -> str:
        """Get the conversation history as text for the compaction LLM call."""
        parts = []
        for msg in self._messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                parts.append(f"[{role}]: {content}")
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    parts.append(f"[tool_call]: {fn.get('name', '?')}({fn.get('arguments', '')})")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all messages (keeps system prompt)."""
        self._messages.clear()


def build_image_content_block(image_url: str) -> dict[str, Any]:
    """Create an image content block for multimodal messages.

    Args:
        image_url: HTTP(S) URL or base64 data URI (data:image/...;base64,...).

    Returns:
        OpenAI-format image content block.

    Raises:
        ValueError: If the URL format is invalid.
    """
    if not image_url:
        msg = "image_url must not be empty"
        raise ValueError(msg)
    if not (
        image_url.startswith("http://")
        or image_url.startswith("https://")
        or image_url.startswith("data:image/")
    ):
        msg = (
            f"Invalid image URL format: must be http(s):// or data:image/ URI, got {image_url[:50]}"
        )
        raise ValueError(msg)
    return {"type": "image_url", "image_url": {"url": image_url}}


def build_multimodal_message(
    text: str,
    images: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a list of content blocks combining text and images.

    Args:
        text: The text portion of the message.
        images: Optional list of image URLs or base64 data URIs.

    Returns:
        List of content blocks suitable for Conversation.add_user_message().
    """
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for url in images or []:
        blocks.append(build_image_content_block(url))
    return blocks
