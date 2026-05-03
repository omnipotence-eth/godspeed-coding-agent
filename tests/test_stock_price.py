"""Tests for godspeed.tools.stock_price."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from godspeed.tools.stock_price import StockPriceTool


class TestStockPriceToolMetadata:
    def test_name(self):
        tool = StockPriceTool()
        assert tool.name == "stock_price"

    def test_risk_level(self):
        tool = StockPriceTool()
        assert tool.risk_level.value == "low"

    def test_description_contains_keywords(self):
        tool = StockPriceTool()
        desc = tool.description.lower()
        assert "stock" in desc

    def test_get_schema(self):
        tool = StockPriceTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "ticker" in schema["properties"]
        assert "period" in schema["properties"]
        assert schema["required"] == ["ticker"]


class TestStockPriceToolExecute:
    @pytest.mark.asyncio
    async def test_missing_ticker(self):
        tool = StockPriceTool()
        result = await tool.execute({}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_empty_ticker(self):
        tool = StockPriceTool()
        result = await tool.execute({"ticker": ""}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_invalid_ticker_type(self):
        tool = StockPriceTool()
        result = await tool.execute({"ticker": 123}, MagicMock())
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_yfinance_exception(self):
        tool = StockPriceTool()
        # Make yfinance.Ticker raise on instantiation.
        # The exception is caught inside _fetch_stock_data and reported
        # as a line in the output, not as is_error=True.
        with patch("yfinance.Ticker", side_effect=Exception):
            result = await tool.execute({"ticker": "FAIL"}, MagicMock())
        assert "Error:" in result.output

    @pytest.mark.asyncio
    async def test_period_option(self):
        tool = StockPriceTool()
        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "Test", "currentPrice": 100.0}
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL", "period": "1mo"}, MagicMock())
        assert result.is_error is False
        mock_ticker.history.assert_called_once()
        call_kwargs = mock_ticker.history.call_args
        assert call_kwargs[1]["period"] == "1mo"

    @pytest.mark.asyncio
    async def test_multiple_tickers(self):
        tool = StockPriceTool()
        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "Test Corp", "currentPrice": 100.0}
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL MSFT"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_ticker_with_spaces(self):
        tool = StockPriceTool()
        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "Test", "currentPrice": 100.0}
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "  AAPL   MSFT  "}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_crypto_ticker(self):
        tool = StockPriceTool()
        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "Bitcoin", "currentPrice": 50000.0}
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "BTC-USD"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_volume_formatting_millions(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test", "currentPrice": 100.0, "volume": 50_000_000}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_volume_formatting_thousands(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test", "currentPrice": 100.0, "volume": 50_000}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_market_cap_trillions(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test", "currentPrice": 100.0, "marketCap": 2.5e12}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_pe_ratio_display(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test", "currentPrice": 100.0, "trailingPE": 25.5}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_52_week_range(self):
        tool = StockPriceTool()
        mock_info = {
            "longName": "Test",
            "currentPrice": 100.0,
            "fiftyTwoWeekLow": 80.0,
            "fiftyTwoWeekHigh": 120.0,
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_dividend_yield(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test", "currentPrice": 100.0, "dividendYield": 0.025}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_no_data_found(self):
        tool = StockPriceTool()
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_hist = MagicMock()
        mock_hist.empty = True

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "FAKE"}, MagicMock())
        assert "FAKE" in result.output

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self):
        tool = StockPriceTool()
        with patch(
            "godspeed.tools.stock_price.StockPriceTool._fetch_stock_data",
            side_effect=Exception("boom"),
        ):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert result.is_error is True
