"""Briefing history tools — let the chat agent query past daily briefings.

These tools run DB queries synchronously via asyncio.run() because LangChain
tool functions are synchronous. The DB functions are async (aiosqlite).
"""

import asyncio
from typing import Any

from langchain_core.tools import Tool


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context.

    LangChain tool functions are sync, but our DB layer is async.
    Uses the existing event loop if available (FastAPI context),
    otherwise creates a new one.
    """
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return loop.run_in_executor(pool, asyncio.run, coro)
    except RuntimeError:
        return asyncio.run(coro)


def _tool_briefing_history(ticker: str, days: str = "7") -> str:
    """Get recent daily briefing analyses for a ticker.

    Returns a formatted summary of recent outlooks, signals, and alerts.
    """
    from api.db import get_briefing_history

    try:
        days_int = int(days)
    except ValueError:
        days_int = 7

    try:
        history = asyncio.run(get_briefing_history(ticker.upper(), days=days_int))
    except RuntimeError:
        return f"Unable to query briefing history for {ticker} (event loop conflict)."

    if not history:
        return f"No briefing history found for {ticker} in the last {days_int} days."

    lines = [f"Briefing history for {ticker.upper()} (last {days_int} days):"]
    lines.append("=" * 50)

    for entry in history:
        date = entry.get("created_at", "Unknown date")
        outlook = entry.get("outlook", "unknown").upper()
        price = entry.get("price", 0)
        change = entry.get("change_pct", 0)
        signal = entry.get("technical_signal", "N/A")
        news = entry.get("news_summary", "N/A")

        lines.append(f"\n{date} — {outlook}")
        lines.append(f"  Price: ${price:.2f} ({change:+.2f}%)")
        lines.append(f"  Signal: {signal}")
        lines.append(f"  News: {news}")

    return "\n".join(lines)


def _tool_latest_briefing(_input: str = "") -> str:
    """Get the most recent full morning briefing summary."""
    from api.db import get_recent_briefings

    try:
        briefings = asyncio.run(get_recent_briefings(limit=1))
    except RuntimeError:
        return "Unable to query latest briefing (event loop conflict)."

    if not briefings:
        return "No briefings have been generated yet."

    b = briefings[0]
    lines = [
        f"Latest Briefing ({b.get('created_at', 'Unknown date')})",
        "=" * 50,
        f"Market Regime: {b.get('market_regime', 'N/A')}",
        f"Positioning: {b.get('market_positioning', 'N/A')}",
        "",
        "Ticker Analysis:",
    ]

    for t in b.get("tickers", []):
        outlook = t.get("outlook", "unknown").upper()
        lines.append(f"  {t['ticker']}: ${t['price']:.2f} ({t['change_pct']:+.2f}%) [{outlook}]")
        lines.append(f"    Signal: {t.get('technical_signal', 'N/A')}")
        lines.append(f"    News: {t.get('news_summary', 'N/A')}")

    alerts = b.get("alerts", "[]")
    if isinstance(alerts, str):
        import json
        try:
            alerts = json.loads(alerts)
        except (json.JSONDecodeError, TypeError):
            alerts = []

    if alerts:
        lines.append("\nAlerts:")
        for alert in alerts:
            lines.append(f"  - {alert}")

    return "\n".join(lines)


def create_briefing_tools(ticker: str) -> list[Tool]:
    """Create briefing history tools bound to a specific ticker.

    Args:
        ticker: Company ticker symbol (used as default for history queries)

    Returns:
        List of LangChain Tool objects for briefing history
    """
    return [
        Tool.from_function(
            name="get_briefing_history",
            description=(
                "Get recent daily briefing analyses for the company. "
                "Returns outlook trends, technical signals, and news summaries "
                "from past morning briefings. Input: number of days to look back (default 7)."
            ),
            func=lambda days="7": _tool_briefing_history(ticker, days),
        ),
        Tool.from_function(
            name="get_latest_briefing",
            description=(
                "Get the most recent full morning briefing summary. "
                "Returns market regime, positioning, all ticker analyses, and alerts."
            ),
            func=_tool_latest_briefing,
        ),
    ]
