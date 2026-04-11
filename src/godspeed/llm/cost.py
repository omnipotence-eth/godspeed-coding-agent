"""Cost estimation for LLM API calls.

Maps model prefixes to per-token pricing. Prices are approximate and updated
periodically — they're for budget awareness, not billing-accurate accounting.
Ollama and local models are always $0.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Pricing: (input_per_million_tokens, output_per_million_tokens)
# Source: provider pricing pages as of April 2026.
# Local/free models are omitted — they default to (0, 0).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o3": (10.0, 40.0),
    "o3-mini": (1.10, 4.40),
    "codex-1": (10.0, 40.0),
    # Google Gemini
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Mistral
    "mistral-large": (2.0, 6.0),
    "mistral-small": (0.10, 0.30),
}

# Provider prefixes that are always free (local inference)
_FREE_PREFIXES = ("ollama/", "ollama_chat/", "lm_studio/", "llamacpp/")


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate the cost of an LLM call in USD.

    Returns 0.0 for local/free models or unknown pricing.
    """
    model_lower = model.lower()

    # Free local models
    if any(model_lower.startswith(prefix) for prefix in _FREE_PREFIXES):
        return 0.0

    # Strip provider prefix for matching (e.g., "anthropic/claude-sonnet..." → "claude-sonnet...")
    name = model_lower.split("/")[-1] if "/" in model_lower else model_lower

    # Find the best matching pricing entry (longest prefix match)
    best_match = ""
    best_pricing = (0.0, 0.0)
    for prefix, pricing in _MODEL_PRICING.items():
        if name.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_pricing = pricing

    if not best_match:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * best_pricing[0]
    output_cost = (output_tokens / 1_000_000) * best_pricing[1]
    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format a cost value for display.

    Returns "free" for zero cost, otherwise "$X.XXXX".
    """
    if cost == 0.0:
        return "free"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
