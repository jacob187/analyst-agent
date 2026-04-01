"""Market-wide context tools.

These tools provide broad market data (indices, macro indicators) that are
not bound to any specific ticker. They help the agent contextualize a
company's performance within the broader market environment.
"""

from langchain_core.tools import Tool

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


def _tool_market_overview() -> str:
    """Return current levels and daily change for major indices and VIX."""
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
    }

    lines = ["Market Overview:"]
    lines.append("=" * 55)

    for name, symbol in indices.items():
        try:
            retriever = YahooFinanceDataRetrieval(symbol)
            hist = retriever.get_historical_prices(period="5d", interval="1d")
            if hist is None or len(hist) < 2:
                lines.append(f"\n{name}: Data unavailable")
                continue

            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change = current - prev
            change_pct = (change / prev) * 100

            direction = "+" if change >= 0 else ""
            lines.append(f"\n{name} ({symbol}):")
            lines.append(f"  Level: {current:,.2f}")
            lines.append(f"  Change: {direction}{change:,.2f} ({direction}{change_pct:.2f}%)")
        except Exception as e:
            lines.append(f"\n{name}: Error fetching data")

    return "\n".join(lines)


def _tool_macro_indicators() -> str:
    """Return key macro indicators: Treasury yields, VIX, and dollar index."""
    indicators = {
        "10-Year Treasury Yield": "^TNX",
        "5-Year Treasury Yield": "^FVX",
        "VIX (Volatility Index)": "^VIX",
        "US Dollar Index": "DX-Y.NYB",
    }

    lines = ["Macro Economic Indicators:"]
    lines.append("=" * 55)

    for name, symbol in indicators.items():
        try:
            retriever = YahooFinanceDataRetrieval(symbol)
            hist = retriever.get_historical_prices(period="1mo", interval="1d")
            if hist is None or hist.empty:
                lines.append(f"\n{name}: Data unavailable")
                continue

            current = float(hist["Close"].iloc[-1])
            first = float(hist["Close"].iloc[0])
            change_1m = ((current - first) / first) * 100

            is_yield = "Yield" in name or "Treasury" in name
            unit = "%" if is_yield else ""

            lines.append(f"\n{name}:")
            lines.append(f"  Current: {current:.2f}{unit}")
            lines.append(f"  1-Month Change: {change_1m:+.2f}%")
        except Exception:
            lines.append(f"\n{name}: Data unavailable")

    return "\n".join(lines)


def create_market_tools() -> list[Tool]:
    """Return market-wide context tools (not ticker-bound)."""
    return [
        Tool.from_function(
            name="get_market_overview",
            description="Get current levels and daily change for S&P 500, Nasdaq, Dow Jones, and VIX. Use for broad market context, benchmark comparison, or answering macro market questions.",
            func=lambda query="": _tool_market_overview(),
        ),
        Tool.from_function(
            name="get_macro_indicators",
            description="Get macro economic indicators: 10-Year and 5-Year Treasury yields, VIX volatility index, and US Dollar Index with 1-month changes. Use for interest rate context and macro environment analysis.",
            func=lambda query="": _tool_macro_indicators(),
        ),
    ]
