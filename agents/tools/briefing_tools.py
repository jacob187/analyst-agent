"""Briefing history tools — let the chat agent query past daily briefings.

The tools are async (``coroutine=``) and await the aiosqlite DB helpers on
the running event loop — the loop that owns the process-wide connection.
"""

from langchain_core.tools import StructuredTool

from agents.tools.tool_args import DEFAULT_BRIEFING_DAYS, DaysArgs, NoArgs


async def _tool_briefing_history(
    ticker: str, user_id: str | None = None, days: int | str = DEFAULT_BRIEFING_DAYS
) -> str:
    """Get recent daily briefing analyses for a ticker.

    Returns a formatted summary of recent outlooks, signals, and alerts.
    """
    from api.db import get_briefing_history

    if not user_id:
        return f"No briefing history available for {ticker} (no user context)."

    try:
        days_int = int(days)
    except (TypeError, ValueError):
        days_int = DEFAULT_BRIEFING_DAYS

    history = await get_briefing_history(ticker.upper(), user_id, days=days_int)

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


async def _tool_latest_briefing(user_id: str | None = None) -> str:
    """Get the most recent full morning briefing summary."""
    from api.db import get_recent_briefings

    briefings = await get_recent_briefings(user_id=user_id, limit=1)

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


def create_briefing_tools(ticker: str, user_id: str | None = None) -> list[StructuredTool]:
    """Create briefing history tools bound to a specific ticker and user.

    Args:
        ticker: Company ticker symbol (used as default for history queries)
        user_id: Anonymous user ID for scoping briefing queries

    Returns:
        List of LangChain StructuredTool objects for briefing history
    """
    return [
        StructuredTool.from_function(
            name="get_briefing_history",
            description=(
                "Get recent daily briefing analyses for the company. "
                "Returns outlook trends, technical signals, and news summaries "
                "from past morning briefings. Input: number of days to look back (default 7)."
            ),
            func=None,
            coroutine=lambda days=DEFAULT_BRIEFING_DAYS: _tool_briefing_history(
                ticker, user_id, days
            ),
            args_schema=DaysArgs,
        ),
        StructuredTool.from_function(
            name="get_latest_briefing",
            description=(
                "Get the most recent full morning briefing summary. "
                "Returns market regime, positioning, all ticker analyses, and alerts."
            ),
            func=None,
            coroutine=lambda: _tool_latest_briefing(user_id),
            args_schema=NoArgs,
        ),
    ]
