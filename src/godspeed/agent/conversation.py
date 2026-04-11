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
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4",
        max_tokens: int = 100_000,
        compaction_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self._system_message: dict[str, Any] = {"role": "system", "content": system_prompt}
        self._messages: list[dict[str, Any]] = []

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

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._messages.append({"role": "user", "content": content})

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
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message."""
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def compact(self, summary: str) -> None:
        """Replace conversation history with a summary.

        Preserves the system prompt and adds the summary as context,
        then clears the message history.
        """
        logger.info(
            "Compacting conversation tokens_before=%d message_count=%d",
            self.token_count,
            len(self._messages),
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
