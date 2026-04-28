"""Token counting for context budget management."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# Model-to-encoding mapping — tiktoken doesn't know non-OpenAI models so we map
# them to the closest tokenizer. cl100k_base (~100k vocab) is a safe default
# since most modern LLMs use similar BPE vocabularies and we only need approximate
# token counts for context budget management, not exact billing.
_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "claude": "cl100k_base",
    "gemini": "cl100k_base",
    "deepseek": "cl100k_base",
    "qwen": "cl100k_base",
    "llama": "cl100k_base",
    "gemma": "cl100k_base",
    "mistral": "cl100k_base",
}
_DEFAULT_ENCODING = "cl100k_base"

IMAGE_BLOCK_TOKEN_ESTIMATE = 765


@lru_cache(maxsize=32)
def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model. Falls back to cl100k_base."""
    model_name = model.split("/")[-1] if "/" in model else model

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.debug("No tiktoken encoding for model=%s, trying prefix mapping", model_name)

    for prefix, encoding in _MODEL_ENCODINGS.items():
        if model_name.startswith(prefix):
            return tiktoken.get_encoding(encoding)

    return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a text string."""
    enc = get_encoding(model)
    return len(enc.encode(text))


def count_message_tokens(messages: list[dict[str, Any]], model: str = "gpt-4") -> int:
    """Estimate token count for a list of chat messages (batch-encoded)."""
    enc = get_encoding(model)
    strings_to_encode: list[str] = []
    image_blocks = 0

    for msg in messages:
        for _, value in msg.items():
            if isinstance(value, str) and value:
                strings_to_encode.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            image_blocks += 1
                            continue
                        for v in item.values():
                            if isinstance(v, str) and v:
                                strings_to_encode.append(v)

    # Batch encode for 3-5x speedup over individual encoding
    token_sum = sum(len(e) for e in enc.encode_ordinary_batch(strings_to_encode))
    # Priming + per-message overhead
    token_sum += 2 + len(messages) * 4
    token_sum += image_blocks * IMAGE_BLOCK_TOKEN_ESTIMATE
    return token_sum
