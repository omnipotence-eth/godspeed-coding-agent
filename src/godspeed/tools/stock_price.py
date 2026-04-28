"""Stock price tool — real-time financial data via Yahoo Finance.

No API key required. Returns current price, daily/historical OHLCV data,
company info, and key statistics for any public ticker.
"""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_HISTORICAL_DAYS_DEFAULT = 5
_MAX_HISTORICAL_DAYS = 90


class StockPriceTool(Tool):
    """Get real-time stock price data and company information.

    Uses Yahoo Finance (yfinance) — no API key required. Returns:
    - Current price (as of last market close if market is closed)
    - Daily data: open, high, low, close, volume
    - Historical OHLCV for N past trading days
    - Company name, sector, market cap, P/E ratio
    - 52-week high/low, dividend yield

    Multiple tickers can be queried at once. Works for stocks, ETFs,
    indices (^GSPC, ^IXIC, ^DJI), and cryptocurrencies (BTC-USD, ETH-USD).
    """

    @property
    def name(self) -> str:
        return "stock_price"

    @property
    def description(self) -> str:
        return (
            "Get real-time stock price data, historical OHLCV, and company info. "
            "No API key required. Works for stocks, ETFs, indices, crypto.\n\n"
            "Examples:\n"
            "  stock_price(ticker='TSLA')\n"
            "  stock_price(ticker='AAPL', period='1mo')\n"
            "  stock_price(ticker='MSFT GOOGL AMZN', period='5d')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Stock ticker symbol (e.g., 'TSLA', 'AAPL', 'MSFT'). "
                        "Space-separate for multiple: 'MSFT GOOGL'. "
                        "Use BTC-USD for Bitcoin, ETH-USD for Ethereum."
                    ),
                    "examples": ["TSLA", "AAPL MSFT GOOGL", "BTC-USD"],
                },
                "period": {
                    "type": "string",
                    "description": (
                        "Data period: '1d' (today only), '5d' (5 days), "
                        "'1mo' (1 month), '3mo' (3 months), '6mo', '1y'. "
                        "Default: '5d'"
                    ),
                    "default": "5d",
                },
            },
            "required": ["ticker"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ticker = arguments.get("ticker", "")
        period = arguments.get("period", "5d")

        if not isinstance(ticker, str) or not ticker.strip():
            return ToolResult.failure("ticker must be a non-empty string")

        symbols = [s.strip().upper() for s in ticker.split() if s.strip()]
        if not symbols:
            return ToolResult.failure("No valid ticker symbols provided")

        try:
            return await self._fetch_stock_data(symbols, period)
        except Exception as exc:
            logger.warning("Stock data fetch failed: %s", exc)
            return ToolResult.failure(f"Could not fetch stock data for {', '.join(symbols)}: {exc}")

    async def _fetch_stock_data(self, symbols: list[str], period: str) -> ToolResult:
        import asyncio

        import yfinance as yf

        lines = []

        for symbol in symbols:
            try:
                ticker_obj = yf.Ticker(symbol)
                info = await asyncio.to_thread(lambda t=ticker_obj: t.info)  # type: ignore[arg-type]
                hist = await asyncio.to_thread(
                    lambda t=ticker_obj, p=period: t.history(period=p)  # type: ignore[arg-type]
                )
            if info.get("dividendYield"):
                extended.append(f"Div Yield: {info['dividendYield'] * 100:.2f}%")
            if extended:
                lines.append("  " + " | ".join(extended))

            # Historical data
            if not hist.empty:
                lines.append("")
                lines.append(f"  Historical ({period}):")
                header = (
                    f"  {'Date':<12} {'Open':>10} {'High':>10} "
                    f"{'Low':>10} {'Close':>10} {'Volume':>12}"
                )
                lines.append(header)
                lines.append(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")
                for date, row in hist.tail(10).iterrows():
                    dt = date.strftime("%Y-%m-%d")
                    vol_str = f"{row['Volume']:,.0f}"
                    if row["Volume"] >= 1_000_000:
                        vol_str = f"{row['Volume'] / 1_000_000:.1f}M"
                    lines.append(
                        f"  {dt:<12} ${row['Open']:>9.2f} ${row['High']:>9.2f} "
                        f"${row['Low']:>9.2f} ${row['Close']:>9.2f} {vol_str:>12}"
                    )

            lines.append("")

        if not lines:
            return ToolResult.success(
                f"No data found for {', '.join(symbols)}. "
                "Check the ticker symbol (use uppercase) or try a different period."
            )

        return ToolResult.success("\n".join(lines))
