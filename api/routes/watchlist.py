"""Watchlist endpoints — manage tracked tickers and generate daily briefings."""

import asyncio
import json
import logging
import os
import re

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from api.db import (
    get_watchlist, get_watchlist_enriched, add_to_watchlist, remove_from_watchlist,
    save_briefing, get_recent_briefings, get_briefing_history,
)
from api.dependencies import ApiKeys, get_api_keys, resolve_model
from api.llm_concurrency import (
    BudgetUsage,
    LLMBudgetExceeded,
    LLMCallCharger,
    LLMConcurrencyExceeded,
    ensure_budget_available,
    llm_slot,
    refusal_message,
)
from api.rate_limit import check_rest_rate_limit, rate_limit_key

router = APIRouter(prefix="/watchlist", tags=["watchlist"])
logger = logging.getLogger("analyst.watchlist")

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")

# Hard ceiling for the full briefing generation (LLM + yfinance fan-out).
# A stuck LLM call would otherwise hold the worker thread, the user's HTTP
# connection, and the model client indefinitely. Env-overridable.
BRIEFING_TIMEOUT_SECONDS = float(os.getenv("BRIEFING_TIMEOUT_SECONDS", "120"))


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
async def get_briefing(
    keys: ApiKeys = Depends(get_api_keys),
    x_model_id: str | None = Header(None),
):
    """Generate AI briefing for the user's watchlist tickers."""
    user_id = keys.require_user_id()
    if not check_rest_rate_limit(
        rate_limit_key(user_id, ip=""),
        bucket="briefing",
        max_calls=5,
        window_seconds=3600,
    ):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")

    tickers_list = await get_watchlist(user_id)
    if not tickers_list:
        raise HTTPException(status_code=400, detail="Watchlist is empty")

    from agents.briefing.briefing_service import BriefingService
    from agents.llm_factory import ThinkingConfig, create_llm

    # Same resolution as chat and filings: an explicit X-Model-Id wins; otherwise
    # the default if the caller can run it, else a model they have a key for.
    # BYOK works for any model; the operator's env key only for lendable ones.
    model, api_key = resolve_model(keys, x_model_id)
    model_id = model.id
    if not api_key:
        provider_display = model.provider.replace("_", " ").title()
        if keys.get_provider_key(model.provider):
            # Only the operator's key resolved, and it won't cover this model.
            detail = f"{model.display_name} needs your own {provider_display} API key. Add it in Settings."
        else:
            detail = f"{provider_display} API key required for {model.display_name}"
        raise HTTPException(status_code=400, detail=detail)

    # Operator-paid: refuse once today's budget is spent, then charge one unit
    # per LLM call actually made (LLMCallCharger below). BYOK requests spend
    # the user's own provider quota and are never charged.
    operator_paid = keys.is_operator_paid(model.provider)
    budget_used = BudgetUsage(user=0, overall=0)
    if operator_paid:
        try:
            budget_used = await ensure_budget_available(user_id)
        except LLMBudgetExceeded as e:
            raise HTTPException(status_code=429, detail=refusal_message(e))

    thinking = ThinkingConfig(enabled=True, level="medium") if model.thinking_capable else None
    llm = create_llm(model.id, api_key, thinking=thinking)

    service = BriefingService(llm, tavily_api_key=keys.tavily_api_key)
    tickers = [t["ticker"] for t in tickers_list]

    # Fetch the user's most recent briefing (if any) so the service can
    # compute a "since last briefing" diff. The new briefing hasn't been
    # saved yet, so "most recent" == "previous."
    prior = await get_recent_briefings(user_id=user_id, limit=1)
    previous_briefing = prior[0] if prior else None

    charger = LLMCallCharger(user_id, used=budget_used) if operator_paid else None
    run_config = {"callbacks": [charger]} if charger else None
    try:
        # BriefingService.generate is sync (yfinance + LLM); offload so it doesn't
        # block other requests' progress on the asyncio loop during the ~10-30s call.
        # Wrap in wait_for so a stuck LLM call surfaces as 504 instead of pinning
        # the worker thread + connection. llm_slot bounds process-wide concurrency.
        async with llm_slot():
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    service.generate, tickers, user_id, model_id, previous_briefing, run_config
                ),
                timeout=BRIEFING_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        logger.warning(
            "Briefing timed out after %ss (tickers=%d)",
            BRIEFING_TIMEOUT_SECONDS,
            len(tickers),
        )
        raise HTTPException(status_code=504, detail="Briefing generation timed out")
    except LLMConcurrencyExceeded:
        raise HTTPException(status_code=503, detail="Server busy, retry shortly")
    except LLMBudgetExceeded as e:
        # The pre-flight passed, but the charger refused the call itself (it
        # costs more units than were left).
        raise HTTPException(status_code=429, detail=refusal_message(e))
    except Exception:
        logger.exception("Briefing generation failed (tickers=%d)", len(tickers))
        raise HTTPException(status_code=500, detail="Briefing generation failed")
    finally:
        # Wait for the charges of the calls made so far, also when the run
        # failed or timed out. After a timeout the worker thread keeps going:
        # a call it starts later is still charged (and a failed charge logged)
        # by the charger, just not awaited here.
        if charger:
            await charger.settle()

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
    ticker: str,
    days: int = Query(30, ge=1, le=365),
    keys: ApiKeys = Depends(get_api_keys),
):
    """Return briefing history for a specific ticker, scoped to the user."""
    user_id = keys.require_user_id()
    history = await get_briefing_history(ticker, user_id, days=days)
    return {"ticker": ticker.upper(), "history": history}
