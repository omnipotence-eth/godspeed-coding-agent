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


class TestStockPriceAdditionalMetrics:
    """Tests covering additional metrics branches not covered by basic tests."""

    @pytest.mark.asyncio
    async def test_regular_market_price(self):
        """Cover regularMarketPrice metric (separate from currentPrice)."""
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "regularMarketPrice": 101.50}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Market:" in result.output

    @pytest.mark.asyncio
    async def test_previous_close(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "previousClose": 99.50}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Prev Close:" in result.output

    @pytest.mark.asyncio
    async def test_open_price(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "open": 100.25}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Open:" in result.output

    @pytest.mark.asyncio
    async def test_day_high_low(self):
        """Cover dayHigh + dayLow (both must be present)."""
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "dayHigh": 105.0, "dayLow": 95.0}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Day:" in result.output

    @pytest.mark.asyncio
    async def test_volume_below_thousand(self):
        """Volume < 1,000 formats with comma separator."""
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "currentPrice": 100.0, "volume": 500}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Vol: 500" in result.output

    @pytest.mark.asyncio
    async def test_market_cap_billions(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "currentPrice": 100.0, "marketCap": 5e9}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "MCap:" in result.output
        assert "B" in result.output

    @pytest.mark.asyncio
    async def test_market_cap_millions(self):
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "currentPrice": 100.0, "marketCap": 500e6}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "MCap:" in result.output
        assert "M" in result.output

    @pytest.mark.asyncio
    async def test_market_cap_below_million(self):
        """Market cap below 1e6 — no MCap line added."""
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "currentPrice": 100.0, "marketCap": 500000}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "MCap:" not in result.output

    @pytest.mark.asyncio
    async def test_sector_and_industry(self):
        """Cover sector + industry formatting in company name."""
        tool = StockPriceTool()
        mock_info = {
            "longName": "Test Corp",
            "sector": "Technology",
            "industry": "Software",
            "currentPrice": 100.0,
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Technology" in result.output
        assert "Software" in result.output

    @pytest.mark.asyncio
    async def test_sector_only_no_industry(self):
        """Sector present but no industry — still formats name."""
        tool = StockPriceTool()
        mock_info = {
            "longName": "Test Corp",
            "sector": "Technology",
            "currentPrice": 100.0,
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Technology" in result.output

    @pytest.mark.asyncio
    async def test_short_name_fallback(self):
        """Fallback to shortName when longName is missing."""
        tool = StockPriceTool()
        mock_info = {"shortName": "ShortCo", "currentPrice": 100.0}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "ShortCo" in result.output


class TestStockPriceHistorical:
    """Tests for historical OHLCV data display."""

    @pytest.mark.asyncio
    async def test_historical_data_with_volume_millions(self):
        """Historical rows with volume >= 1,000,000 display in M format."""
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "currentPrice": 100.0}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        mock_date = MagicMock()
        mock_date.strftime.return_value = "2026-05-09"
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = (
            lambda k: {
                "Open": 150.0,
                "High": 155.0,
                "Low": 149.0,
                "Close": 152.0,
                "Volume": 2_500_000,
            }[k]
        )

        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.tail.return_value.iterrows.return_value = [(mock_date, mock_row)]
        mock_ticker.history.return_value = mock_hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Historical" in result.output
        assert "2026-05-09" in result.output
        assert "2.5M" in result.output

    @pytest.mark.asyncio
    async def test_historical_data_with_volume_thousands(self):
        """Historical rows with volume between 1K-1M display in K format via raw number."""
        tool = StockPriceTool()
        mock_info = {"longName": "Test Corp", "currentPrice": 100.0}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        mock_date = MagicMock()
        mock_date.strftime.return_value = "2026-05-09"
        mock_row = MagicMock()
        # Volume 10,000 — the historical formatting checks >= 1_000_000, so this
        # falls through to the raw comma-formatted number path.
        mock_row.__getitem__.side_effect = (
            lambda k: {
                "Open": 150.0,
                "High": 155.0,
                "Low": 149.0,
                "Close": 152.0,
                "Volume": 10_000,
            }[k]
        )

        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.tail.return_value.iterrows.return_value = [(mock_date, mock_row)]
        mock_ticker.history.return_value = mock_hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "AAPL"}, MagicMock())
        assert not result.is_error
        assert "Historical" in result.output

    @pytest.mark.asyncio
    async def test_no_data_all_symbols(self):
        """When all symbols produce no data, return helpful message."""
        tool = StockPriceTool()
        mock_info: dict = {}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_hist = MagicMock()
        mock_hist.empty = True
        mock_ticker.history.return_value = mock_hist
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await tool.execute({"ticker": "GHOST"}, MagicMock())
        assert not result.is_error
        assert "GHOST" in result.output

    @pytest.mark.asyncio
    async def test_fetch_stock_data_empty_symbols(self):
        """Direct call to _fetch_stock_data with empty symbols list."""
        tool = StockPriceTool()
        result = await tool._fetch_stock_data([], "5d")
        assert not result.is_error
        assert "No data found" in result.output
