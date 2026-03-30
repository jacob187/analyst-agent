"""Watchlist endpoints — manage tracked tickers and generate daily briefings."""

import re

from fastapi import APIRouter, HTTPException

from api.db import get_watchlist, add_to_watchlist, remove_from_watchlist, get_settings

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")


@router.get("")
async def list_watchlist():
    """Return all tickers in the watchlist."""
    tickers = await get_watchlist()
    return {"tickers": tickers}


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
    return {"success": True, "ticker": ticker}


@router.delete("/{ticker}")
async def remove_ticker(ticker: str):
    """Remove a ticker from the watchlist."""
    removed = await remove_from_watchlist(ticker)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{ticker.upper()} not in watchlist")
    return {"success": True}


@router.get("/briefing")
async def get_briefing():
    """Generate AI briefing for all watchlist tickers."""
    tickers_list = await get_watchlist()
    if not tickers_list:
        raise HTTPException(status_code=400, detail="Watchlist is empty")

    settings = await get_settings()
    if not settings or not settings.get("google_api_key"):
        raise HTTPException(status_code=400, detail="Google API key not configured")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from agents.briefing.briefing_service import BriefingService

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings["google_api_key"],
    )

    service = BriefingService(llm)
    tickers = [t["ticker"] for t in tickers_list]

    try:
        briefing = service.generate(tickers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Briefing generation failed: {e}")

    return {"briefing": briefing, "tickers": tickers}
