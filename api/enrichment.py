"""Lazy company enrichment — fills name and sector from Yahoo Finance on first access."""

from api.db import get_company, update_company


async def enrich_company(ticker: str) -> dict | None:
    """Enrich a company record with name and sector from yfinance.

    Only fetches from yfinance if name is not already populated (cached in DB).
    Returns the company dict (enriched or existing), or None if not tracked.
    """
    company = await get_company(ticker)
    if company is None:
        return None

    # Already enriched — skip the network call
    if company["name"] is not None:
        return company

    try:
        from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval

        retriever = YahooFinanceDataRetrieval(ticker)
        info = retriever.ticker.info

        name = info.get("shortName") or info.get("longName")
        sector = info.get("sector")

        if name or sector:
            await update_company(ticker, name=name, sector=sector)
            company["name"] = name
            company["sector"] = sector
    except Exception:
        pass  # Enrichment is best-effort

    return company
