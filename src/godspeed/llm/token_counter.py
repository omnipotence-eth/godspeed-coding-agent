"""Token counting for context budget management."""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# Model-to-encoding mapping for common models
_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
}
_DEFAULT_ENCODING = "cl100k_base"


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model. Falls back to cl100k_base."""
    # Strip provider prefix (e.g., "anthropic/claude-..." -> "claude-...")
    model_name = model.split("/")[-1] if "/" in model else model

    # Try tiktoken's built-in model lookup
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        pass

    # Check our mapping
    for prefix, encoding in _MODEL_ENCODINGS.items():
        if model_name.startswith(prefix):
            return tiktoken.get_encoding(encoding)

    # Default
    return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a text string."""
    enc = get_encoding(model)
    return len(enc.encode(text))


def count_message_tokens(messages: list[dict[str, Any]], model: str = "gpt-4") -> int:
    """Estimate token count for a list of chat messages.

    Uses the OpenAI message format: each message has overhead tokens
    for role/name separators, plus the content tokens.
    """
    enc = get_encoding(model)
    tokens = 0
    for msg in messages:
        # Per-message overhead (~4 tokens: role, content separators)
        tokens += 4
        for _key, value in msg.items():
            if isinstance(value, str):
                tokens += len(enc.encode(value))
            elif isinstance(value, list):
                # Tool calls or content blocks
                for item in value:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str):
                                tokens += len(enc.encode(v))
    # Priming tokens
    tokens += 2
    return tokens
