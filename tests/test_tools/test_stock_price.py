"""Tests for the stock_price tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from godspeed.tools.base import ToolContext
from godspeed.tools.stock_price import StockPriceTool


class TestStockPriceTool:
    def test_name_and_risk(self) -> None:
        tool = StockPriceTool()
        assert tool.name == "stock_price"
        assert tool.risk_level.value == "low"

    def test_schema_has_ticker(self) -> None:
        tool = StockPriceTool()
        schema = tool.get_schema()
        assert "ticker" in schema["properties"]
        assert schema["required"] == ["ticker"]

    def test_missing_ticker(self) -> None:
        tool = StockPriceTool()
        ctx = ToolContext(cwd=Path.cwd(), session_id="test")
        result = asyncio.run(tool.execute({}, ctx))
        assert result.is_error
        assert "ticker" in result.error.lower()

    def test_empty_ticker(self) -> None:
        tool = StockPriceTool()
        ctx = ToolContext(cwd=Path.cwd(), session_id="test")
        result = asyncio.run(tool.execute({"ticker": "   "}, ctx))
        assert result.is_error
