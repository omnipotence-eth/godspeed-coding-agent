"""Tests for retrieval sub-agent tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from godspeed.agent.retrieval_agent import RETRIEVAL_SYSTEM_PROMPT, RetrievalSubAgentTool
from godspeed.tools.base import RiskLevel, ToolResult


class TestRetrievalSubAgentTool:
    """Test RetrievalSubAgentTool metadata and execution."""

    def test_tool_metadata(self) -> None:
        coordinator = MagicMock()
        tool = RetrievalSubAgentTool(coordinator)
        assert tool.name == "retrieval"
        assert tool.risk_level == RiskLevel.READ_ONLY
        assert tool.description

    def test_tool_schema(self) -> None:
        coordinator = MagicMock()
        tool = RetrievalSubAgentTool(coordinator)
        schema = tool.get_schema()
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    @pytest.mark.asyncio
    async def test_execute_delegates_to_coordinator(self) -> None:
        coordinator = AsyncMock()
        coordinator.spawn_retrieval = AsyncMock(return_value="file:a.py:1-10")
        tool = RetrievalSubAgentTool(coordinator)

        result = await tool.execute({"query": "find foo"}, MagicMock())
        assert isinstance(result, ToolResult)
        assert "file:a.py:1-10" in result.output

    @pytest.mark.asyncio
    async def test_execute_no_query_returns_failure(self) -> None:
        coordinator = MagicMock()
        tool = RetrievalSubAgentTool(coordinator)

        result = await tool.execute({}, MagicMock())
        assert result.is_error
        assert "query is required" in result.error

    def test_retrieval_system_prompt_has_read_only_instruction(self) -> None:
        assert "read-only" in RETRIEVAL_SYSTEM_PROMPT.lower()
        assert "file:" in RETRIEVAL_SYSTEM_PROMPT
