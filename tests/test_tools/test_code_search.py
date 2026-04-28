"""Tests for the code search tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godspeed.tools.base import RiskLevel, ToolContext
from godspeed.tools.code_search import CodeSearchTool


@pytest.fixture()
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session")


class TestCodeSearchTool:
    """Test CodeSearchTool."""

    def test_name_and_risk(self) -> None:
        index = MagicMock()
        tool = CodeSearchTool(index)
        assert tool.name == "code_search"
        assert tool.risk_level == RiskLevel.READ_ONLY

    def test_schema_has_query(self) -> None:
        index = MagicMock()
        tool = CodeSearchTool(index)
        schema = tool.get_schema()
        assert "query" in schema["properties"]

    @pytest.mark.asyncio()
    async def test_missing_query(self, context: ToolContext) -> None:
        index = MagicMock()
        tool = CodeSearchTool(index)
        result = await tool.execute({}, context)
        assert result.is_error

    @pytest.mark.asyncio()
    async def test_unavailable_index(self, context: ToolContext) -> None:
        index = MagicMock()
        index.is_available = False
        tool = CodeSearchTool(index)
        result = await tool.execute({"query": "test"}, context)
        assert result.is_error
        assert "not available" in (result.error or "")

    @pytest.mark.asyncio()
    async def test_building_index(self, context: ToolContext) -> None:
        index = MagicMock()
        index.is_available = True
        index.is_building = True
        tool = CodeSearchTool(index)
        result = await tool.execute({"query": "test"}, context)
        assert not result.is_error
        assert "being built" in result.output

    @pytest.mark.asyncio()
    async def test_no_results(self, context: ToolContext) -> None:
        index = MagicMock()
        index.is_available = True
        index.is_building = False
        index.search.return_value = []
        tool = CodeSearchTool(index)
        result = await tool.execute({"query": "test"}, context)
        assert "No results" in result.output

    @pytest.mark.asyncio()
    async def test_with_results(self, context: ToolContext) -> None:
        from godspeed.context.codebase_index import SearchResult

        index = MagicMock()
        index.is_available = True
        index.is_building = False
        index.search.return_value = [
            SearchResult(
                file_path="src/main.py",
                start_line=10,
                end_line=20,
                content="def hello():\n    return 'world'",
                score=0.95,
            ),
        ]
        tool = CodeSearchTool(index)
        result = await tool.execute({"query": "hello function"}, context)
        assert not result.is_error
        assert "main.py" in result.output
        assert "0.95" in result.output
        assert "hello" in result.output

    @pytest.mark.asyncio()
    async def test_custom_top_k(self, context: ToolContext) -> None:
        index = MagicMock()
        index.is_available = True
        index.is_building = False
        index.search.return_value = []
        tool = CodeSearchTool(index)
        await tool.execute({"query": "test", "top_k": 3}, context)
        index.search.assert_called_once_with("test", top_k=3)
