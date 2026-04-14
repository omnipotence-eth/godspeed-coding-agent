"""Tests for LLM cost estimation and model selection."""

from __future__ import annotations

from godspeed.llm.cost import estimate_cost, format_cost, get_cheapest_model


class TestEstimateCost:
    def test_ollama_free(self) -> None:
        assert estimate_cost("ollama/qwen3:4b", 100_000, 50_000) == 0.0

    def test_ollama_chat_free(self) -> None:
        assert estimate_cost("ollama_chat/qwen3:4b", 100_000, 50_000) == 0.0

    def test_claude_sonnet(self) -> None:
        cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        # $3/M input + $15/M output = $18
        assert cost == 18.0

    def test_gpt_4o(self) -> None:
        cost = estimate_cost("gpt-4o-2024-08-06", 1_000_000, 1_000_000)
        # $2.50/M input + $10/M output = $12.50
        assert cost == 12.5

    def test_unknown_model_free(self) -> None:
        assert estimate_cost("some-unknown-model", 100_000, 50_000) == 0.0

    def test_with_provider_prefix(self) -> None:
        cost = estimate_cost("anthropic/claude-sonnet-4-20250514", 1_000_000, 0)
        assert cost == 3.0  # $3/M input

    def test_zero_tokens(self) -> None:
        assert estimate_cost("claude-sonnet", 0, 0) == 0.0

    def test_small_token_count(self) -> None:
        cost = estimate_cost("claude-sonnet", 1000, 500)
        assert cost > 0
        assert cost < 0.02  # ~$0.0105 for 1K input + 500 output


class TestGetCheapestModel:
    def test_empty_list(self) -> None:
        assert get_cheapest_model([]) == ""

    def test_ollama_is_cheapest(self) -> None:
        models = ["claude-sonnet-4-20250514", "ollama/qwen3:4b", "gpt-4o"]
        assert get_cheapest_model(models) == "ollama/qwen3:4b"

    def test_all_api_models(self) -> None:
        models = ["claude-opus-4-20250514", "claude-haiku-3.5", "gpt-4o"]
        result = get_cheapest_model(models)
        assert result == "claude-haiku-3.5"  # Cheapest API model

    def test_single_model(self) -> None:
        assert get_cheapest_model(["gpt-4o"]) == "gpt-4o"

    def test_all_ollama(self) -> None:
        models = ["ollama/qwen3:4b", "ollama/gemma3:12b"]
        # First free model wins
        assert get_cheapest_model(models) == "ollama/qwen3:4b"


class TestFormatCost:
    def test_free(self) -> None:
        assert format_cost(0.0) == "free"

    def test_small_cost(self) -> None:
        assert format_cost(0.0042) == "$0.0042"

    def test_normal_cost(self) -> None:
        assert format_cost(1.50) == "$1.50"
