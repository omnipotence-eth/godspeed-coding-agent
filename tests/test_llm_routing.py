"""Tests for model routing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from godspeed.llm.client import ChatResponse, LLMClient, ModelRouter


class TestModelRouter:
    """Test model routing by task type."""

    def test_no_routing_returns_default(self) -> None:
        router = ModelRouter()
        assert router.route("gpt-4o") == "gpt-4o"

    def test_no_task_type_returns_default(self) -> None:
        router = ModelRouter(routing={"plan": "claude-sonnet-4"})
        assert router.route("gpt-4o") == "gpt-4o"

    def test_routes_plan_to_configured_model(self) -> None:
        router = ModelRouter(routing={"plan": "claude-sonnet-4"})
        assert router.route("ollama/qwen3:4b", task_type="plan") == "claude-sonnet-4"

    def test_routes_edit_to_configured_model(self) -> None:
        router = ModelRouter(routing={"edit": "ollama/qwen3:4b"})
        assert router.route("gpt-4o", task_type="edit") == "ollama/qwen3:4b"

    def test_unmatched_task_type_returns_default(self) -> None:
        router = ModelRouter(routing={"plan": "claude-sonnet-4"})
        assert router.route("gpt-4o", task_type="chat") == "gpt-4o"

    def test_has_routing_true(self) -> None:
        router = ModelRouter(routing={"plan": "claude-sonnet-4"})
        assert router.has_routing is True

    def test_has_routing_false(self) -> None:
        router = ModelRouter()
        assert router.has_routing is False

    def test_routes_property(self) -> None:
        routing = {"plan": "model-a", "edit": "model-b"}
        router = ModelRouter(routing=routing)
        assert router.routes == routing
        # Should be a copy
        router.routes["new"] = "model-c"
        assert "new" not in router._routing


class TestLLMClientRouting:
    """Test that LLMClient uses model routing."""

    @pytest.mark.asyncio
    async def test_chat_with_task_type_routes(self) -> None:
        """Chat with task_type should use the routed model."""
        router = ModelRouter(routing={"plan": "claude-sonnet-4"})
        client = LLMClient(model="gpt-4o", router=router)

        response = ChatResponse(
            content="Plan result",
            tool_calls=[],
            finish_reason="stop",
            usage={},
        )
        client._chat_with_fallback = AsyncMock(return_value=response)

        result = await client.chat(
            messages=[{"role": "user", "content": "Plan this"}],
            task_type="plan",
        )
        assert result.content == "Plan result"
        # Model should have been temporarily changed to claude-sonnet-4
        # but restored after the call
        assert client.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_without_task_type_uses_default(self) -> None:
        """Chat without task_type should use the default model."""
        router = ModelRouter(routing={"plan": "claude-sonnet-4"})
        client = LLMClient(model="gpt-4o", router=router)

        response = ChatResponse(
            content="Default result",
            tool_calls=[],
            finish_reason="stop",
            usage={},
        )
        client._chat_with_fallback = AsyncMock(return_value=response)

        result = await client.chat(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result.content == "Default result"
        assert client.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_routing_restores_model_on_error(self) -> None:
        """Model should be restored even if the routed call fails."""
        router = ModelRouter(routing={"plan": "bad-model"})
        client = LLMClient(model="gpt-4o", router=router)
        client._chat_with_fallback = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await client.chat(
                messages=[{"role": "user", "content": "Plan"}],
                task_type="plan",
            )
        # Model should be restored
        assert client.model == "gpt-4o"

    def test_default_router_is_noop(self) -> None:
        """LLMClient without explicit router should have a no-op router."""
        client = LLMClient(model="gpt-4o")
        assert isinstance(client.router, ModelRouter)
        assert not client.router.has_routing
