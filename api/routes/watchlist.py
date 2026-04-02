"""Watchlist endpoints — manage tracked tickers and generate daily briefings."""

import json
import os
import re

from fastapi import APIRouter, HTTPException, Header

from api.db import (
    get_watchlist, add_to_watchlist, remove_from_watchlist,
    save_briefing, get_recent_briefings, get_briefing_history,
)

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")


@router.get("")
async def list_watchlist():
    """Return all tickers in the watchlist, enriched with company name/sector."""
    from api.db import get_company

    tickers = await get_watchlist()

    # Enrich each ticker with company metadata (name, sector) if available
    enriched = []
    for t in tickers:
        company = await get_company(t["ticker"])
        entry = {**t}
        if company:
            entry["name"] = company.get("name")
            entry["sector"] = company.get("sector")
        enriched.append(entry)

    return {"tickers": enriched}


@router.post("")
async def add_ticker(data: dict):
    """Add a ticker to the watchlist. Body: {"ticker": "AAPL"}"""
    ticker = data.get("ticker", "").strip().upper()
    if not ticker or not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    added = await add_to_watchlist(ticker)
    if not added:
        raise HTTPException(
            status_code=409,
            detail="Ticker already in watchlist or watchlist full (max 10)"
        )

    # Best-effort enrichment — fetch company name/sector in background
    try:
        from api.enrichment import enrich_company
        await enrich_company(ticker)
    except Exception:
        pass

    return {"success": True, "ticker": ticker}


@router.delete("/{ticker}")
async def remove_ticker(ticker: str):
    """Remove a ticker from the watchlist."""
    removed = await remove_from_watchlist(ticker)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{ticker.upper()} not in watchlist")
    return {"success": True}


@router.get("/briefing")
async def get_briefing(
    x_google_api_key: str | None = Header(None),
    x_tavily_api_key: str | None = Header(None),
):
    """Generate AI briefing for all watchlist tickers.

    API keys are passed via headers (from browser localStorage) with env var
    fallback for local development.
    """
    tickers_list = await get_watchlist()
    if not tickers_list:
        raise HTTPException(status_code=400, detail="Watchlist is empty")

    google_api_key = x_google_api_key or os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=400, detail="Google API key required (X-Google-Api-Key header)")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from agents.briefing.briefing_service import BriefingService

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=google_api_key,
        temperature=0,
        thinking_level="medium",
        include_thoughts=True,
    )

    tavily_key = x_tavily_api_key or os.getenv("TAVILY_API_KEY")
    service = BriefingService(llm, tavily_api_key=tavily_key)
    tickers = [t["ticker"] for t in tickers_list]

    try:
        result = service.generate(tickers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Briefing generation failed: {e}")

    # Persist to database (non-blocking — don't fail the response if DB write fails)
    try:
        analysis = result.analysis
        await save_briefing(
            raw_json=analysis.model_dump_json(),
            market_regime=analysis.market_regime,
            market_positioning=analysis.market_positioning,
            alerts_json=json.dumps(analysis.alerts),
            thinking=result.thinking or None,
            tickers=[t.model_dump() for t in analysis.tickers],
        )
    except Exception:
        pass  # DB persistence is best-effort; don't break the response

    return {
        "briefing": result.analysis.to_markdown(),
        "thinking": result.thinking,
        "structured": result.analysis.model_dump(),
        "tickers": tickers,
    }


@router.get("/briefing/history")
async def briefing_history():
    """Return recent briefings."""
    briefings = await get_recent_briefings(limit=20)
    return {"briefings": briefings}


@router.get("/briefing/history/{ticker}")
async def briefing_history_by_ticker(ticker: str, days: int = 30):
    """Return briefing history for a specific ticker."""
    history = await get_briefing_history(ticker, days=days)
    return {"ticker": ticker.upper(), "history": history}
