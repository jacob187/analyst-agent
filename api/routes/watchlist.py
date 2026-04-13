"""Watchlist endpoints — manage tracked tickers and generate daily briefings."""

import json
import re

from fastapi import APIRouter, Depends, HTTPException

from api.db import (
    get_watchlist, get_watchlist_enriched, add_to_watchlist, remove_from_watchlist,
    save_briefing, get_recent_briefings, get_briefing_history,
)
from api.dependencies import ApiKeys, get_api_keys

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")


@router.get("")
async def list_watchlist(keys: ApiKeys = Depends(get_api_keys)):
    """Return tickers in the user's watchlist, enriched with company name/sector."""
    user_id = keys.require_user_id()
    enriched = await get_watchlist_enriched(user_id)
    return {"tickers": enriched}


@router.post("")
async def add_ticker(data: dict, keys: ApiKeys = Depends(get_api_keys)):
    """Add a ticker to the user's watchlist. Body: {"ticker": "AAPL"}"""
    user_id = keys.require_user_id()
    ticker = data.get("ticker", "").strip().upper()
    if not ticker or not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    added = await add_to_watchlist(ticker, user_id)
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
async def remove_ticker(ticker: str, keys: ApiKeys = Depends(get_api_keys)):
    """Remove a ticker from the user's watchlist."""
    user_id = keys.require_user_id()
    removed = await remove_from_watchlist(ticker, user_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{ticker.upper()} not in watchlist")
    return {"success": True}


@router.get("/briefing")
async def get_briefing(keys: ApiKeys = Depends(get_api_keys)):
    """Generate AI briefing for the user's watchlist tickers."""
    user_id = keys.require_user_id()
    tickers_list = await get_watchlist(user_id)
    if not tickers_list:
        raise HTTPException(status_code=400, detail="Watchlist is empty")

    from agents.briefing.briefing_service import BriefingService
    from agents.llm_factory import ThinkingConfig, create_llm
    from agents.model_registry import get_default_model, get_model

    model_id = keys.model_id or get_default_model().id
    model = get_model(model_id) or get_default_model()

    api_key = keys.get_provider_key(model.provider)
    if not api_key:
        provider_display = model.provider.replace("_", " ").title()
        raise HTTPException(status_code=400, detail=f"{provider_display} API key required for {model.display_name}")

    thinking = ThinkingConfig(enabled=True, level="medium") if model.thinking_capable else None
    llm = create_llm(model.id, api_key, thinking=thinking)

    service = BriefingService(llm, tavily_api_key=keys.tavily_api_key)
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
            user_id=user_id,
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
async def briefing_history(keys: ApiKeys = Depends(get_api_keys)):
    """Return recent briefings for this user."""
    user_id = keys.require_user_id()
    briefings = await get_recent_briefings(user_id=user_id, limit=20)
    return {"briefings": briefings}


@router.get("/briefing/history/{ticker}")
async def briefing_history_by_ticker(
    ticker: str, days: int = 30, keys: ApiKeys = Depends(get_api_keys)
):
    """Return briefing history for a specific ticker, scoped to the user."""
    user_id = keys.require_user_id()
    history = await get_briefing_history(ticker, user_id, days=days)
    return {"ticker": ticker.upper(), "history": history}
