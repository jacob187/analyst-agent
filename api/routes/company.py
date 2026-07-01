"""Company dashboard endpoints — profile and filing analysis.

Profile endpoint aggregates yfinance, technicals, patterns, and market regime
into a single JSON response with no LLM calls (< 1s).

Filings endpoint runs LLM analysis on SEC filings with DB-first caching:
first load is slow (LLM calls), every subsequent load is instant (SQLite).
Cache invalidation is automatic — keyed by accession number, so new filings
trigger fresh analysis without TTL.
"""

import asyncio
import json
import logging
import time
from typing import Any

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.dependencies import ApiKeys, get_api_keys
from api.llm_concurrency import (
    LLMBudgetExceeded,
    check_and_charge_budget,
    llm_slot,
)
from api.rate_limit import check_rest_rate_limit, rate_limit_key
from api.validators import TICKER_RE

logger = logging.getLogger("analyst.filings")

router = APIRouter(prefix="/api/company")

# Server-side cache: ticker -> JSON-round-tripped payload dict.
# Bounded by cachetools to prevent unbounded growth under ticker rotation.
_PROFILE_CACHE_TTL = 300  # 5 minutes (also used for Cache-Control max-age)
_profile_cache: TTLCache = TTLCache(maxsize=1024, ttl=_PROFILE_CACHE_TTL)

# Bounded per-ticker lock store. TTL-based eviction prevents unbounded growth
# under ticker rotation — once a ticker hasn't been requested for the TTL
# window, its lock is dropped from the cache. Any in-flight request that
# already pulled the Lock onto its stack keeps the object alive; eviction
# only means the *next* unrelated request for the same ticker creates a fresh
# lock. Safe because the lock is a per-ticker dedup gate, not a global mutex.
_profile_locks: TTLCache = TTLCache(maxsize=4096, ttl=600)

# Per-IP concurrency limiter for SSE streams.
# Each stream holds up to 6 LLM calls — capping concurrent streams per IP
# prevents a single client from exhausting the thread pool or burning API credits.
_MAX_CONCURRENT_STREAMS = 2

# Bounded per-IP semaphore store. Same eviction semantics as `_profile_locks`.
# Replaces the prior `defaultdict(lambda: Semaphore(...))` which grew without
# bound — every novel IP added a Semaphore that was never reclaimed.
_stream_semaphores: TTLCache = TTLCache(maxsize=4096, ttl=600)


def _get_stream_semaphore(client_ip: str) -> asyncio.Semaphore:
    """Return the per-IP stream semaphore, creating one on first sight.

    Wraps the TTLCache lookup so callers don't repeat the get-or-create dance.
    """
    sem = _stream_semaphores.get(client_ip)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)
        _stream_semaphores[client_ip] = sem
    return sem


def _get_profile_lock(ticker: str) -> asyncio.Lock:
    """Return the per-ticker profile lock, creating one on first sight."""
    lock = _profile_locks.get(ticker)
    if lock is None:
        lock = asyncio.Lock()
        _profile_locks[ticker] = lock
    return lock


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


async def _build_profile(ticker: str) -> dict[str, Any]:
    """Async profile assembly. Fans out yfinance calls in parallel."""
    from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
    from agents.technical_workflow.process_technical_indicators import TechnicalIndicators
    from agents.technical_workflow.pattern_recognition import PatternRecognitionEngine
    from agents.market_analysis.regime_detector import MarketRegimeDetector

    retriever = YahooFinanceDataRetrieval(ticker)

    # Fan out 5 independent network fetches in parallel.
    company, quote, earnings, df, regime = await asyncio.gather(
        asyncio.to_thread(_safe, retriever.get_company_profile, {}),
        asyncio.to_thread(_safe, retriever.get_live_price, {}),
        asyncio.to_thread(_safe, retriever.get_earnings_calendar, {}),
        asyncio.to_thread(_safe, lambda: retriever.get_historical_prices(period="1y", interval="1d"), None),
        asyncio.to_thread(_safe, lambda: MarketRegimeDetector().detect_regime(), {}),
    )

    # Technical indicators + patterns from the OHLCV frame. CPU-bound;
    # run on a worker thread so the event loop stays free.
    def _compute_ta(frame):
        technicals: dict[str, Any] = {}
        patterns: list[dict[str, Any]] = []
        if frame is None or frame.empty:
            return technicals, patterns
        ti = TechnicalIndicators(ticker)
        all_ind = ti.calculate_all_indicators(frame)
        technicals = {
            "rsi": all_ind.get("rsi", {}),
            "macd": all_ind.get("macd", {}),
            "adx": all_ind.get("adx", {}),
            "bollinger_bands": all_ind.get("bollinger_bands", {}),
            "moving_averages": all_ind.get("moving_averages", {}),
            "volatility": all_ind.get("volatility", {}),
        }
        try:
            engine = PatternRecognitionEngine()
            raw = engine.detect_all_patterns(frame)
            patterns = [
                {
                    "type": p.get("type", ""),
                    "direction": p.get("direction", ""),
                    "confidence": round(p.get("confidence", 0), 2),
                    "status": p.get("status", ""),
                }
                for p in raw
            ]
        except Exception:
            pass
        return technicals, patterns

    technicals, patterns = await asyncio.to_thread(_compute_ta, df)

    # Sanitise external URL — only allow http(s) schemes to prevent
    # javascript: URI injection when the frontend renders the link.
    raw_website = company.get("website") or ""
    from urllib.parse import urlparse
    safe_website = raw_website if urlparse(raw_website).scheme in ("http", "https") else None

    # Split company dict into "company" (metadata) and "metrics" (financials)
    company_metadata = {
        "name": company.get("shortName"),
        "sector": company.get("sector"),
        "industry": company.get("industry"),
        "country": company.get("country"),
        "website": safe_website,
        "summary": company.get("longBusinessSummary"),
        "employees": company.get("fullTimeEmployees"),
    }
    metrics = {
        "market_cap": company.get("marketCap"),
        "pe_ratio": company.get("trailingPE"),
        "forward_pe": company.get("forwardPE"),
        "price_to_book": company.get("priceToBook"),
        "52wk_high": company.get("fiftyTwoWeekHigh"),
        "52wk_low": company.get("fiftyTwoWeekLow"),
        "dividend_yield": company.get("dividendYield"),
        "beta": company.get("beta"),
    }

    # Round-trip through JSON inside the thread (not on the event loop)
    # to guarantee serializability — numpy floats, Timestamps, etc.
    raw = {
        "ticker": ticker,
        "company": company_metadata,
        "metrics": metrics,
        "quote": quote,
        "earnings": earnings,
        "technicals": technicals,
        "patterns": patterns,
        "regime": regime,
    }
    # Round-trip through JSON off the event loop to coerce numpy floats,
    # Timestamps, etc. to JSON-safe primitives.
    return await asyncio.to_thread(lambda: json.loads(json.dumps(raw, default=str)))


@router.get("/{ticker}/profile")
async def get_company_profile(ticker: str, keys: ApiKeys = Depends(get_api_keys)):
    """Return aggregated company profile for the dashboard Overview tab.

    Combines yfinance metadata, live quote, technical indicators, chart
    patterns, and market regime into a single response. No LLM calls —
    pure data aggregation, typically responds in < 1 second.
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ticker = ticker.upper()

    # Per-ticker lock prevents duplicate in-flight fetches for the same ticker.
    lock = _get_profile_lock(ticker)
    async with lock:
        # Check server-side cache (inside lock to avoid thundering herd)
        cached = _profile_cache.get(ticker)
        if cached is not None:
            return JSONResponse(
                content=cached,
                headers={"Cache-Control": f"public, max-age={_PROFILE_CACHE_TTL}"},
            )

        # _build_profile fans out yfinance calls in parallel via asyncio.gather.
        payload = await _build_profile(ticker)
        _profile_cache[ticker] = payload

        # Track the company as soon as its profile is first loaded.
        # This is the earliest signal that a user is interested in a ticker —
        # before any chat session or filings analysis is triggered.
        from api.db import ensure_company, track_company_view
        if keys.user_id:
            await track_company_view(ticker, keys.user_id)
        else:
            await ensure_company(ticker)

    return JSONResponse(
        content=payload,
        headers={"Cache-Control": f"public, max-age={_PROFILE_CACHE_TTL}"},
    )


# =============================================================================
# Filings endpoint — LLM-powered SEC analysis with DB-first caching
# =============================================================================

def _sse(data: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _build_edgar_url(cik: str, accession: str) -> str:
    """Construct a URL to the filing on SEC EDGAR.

    EDGAR URLs follow the pattern:
    https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/
    where accession_no_dashes strips the hyphens from the accession number.

    Returns empty string if cik or accession is missing.
    """
    if not cik or not accession:
        return ""
    cleaned = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{cleaned}/"


def _fetch_filing_data(ticker: str, fetch_raw: bool = True) -> dict[str, Any]:
    """Synchronous SEC data fetching — runs in a thread pool.

    Returns filing metadata plus, when ``fetch_raw`` is True, the raw section
    text needed for LLM analysis. Keyless callers (who can only ever be served
    *cached* analyses) pass ``fetch_raw=False`` to skip the expensive
    per-section extraction/parse — only the metadata (accession, used as the
    cache key) is fetched. Each filing type fails independently.

    For foreign private issuers (BP, BABA, TSM), falls back to 20-F when
    no 10-K is available. 6-K is NOT a 10-Q replacement — edgartools' SixK
    class is a press-release wrapper without structured section access.
    """
    from agents.sec_workflow.get_SEC_data import SECDataRetrieval

    retriever = SECDataRetrieval(ticker)
    result: dict[str, Any] = {"ticker": ticker}

    # --- 10-K (or 20-F for foreign filers) ---
    try:
        retriever.get_tenk_filing()
        meta = retriever._tenk_metadata
        if meta:
            tenk: dict[str, Any] = {
                "metadata": {
                    **meta.to_dict(),
                    "edgar_url": _build_edgar_url(meta.cik, meta.accession),
                    "form_type": "10-K",
                },
            }
            if fetch_raw:
                tenk.update({
                    "risk_raw": retriever.get_risk_factors_raw("10-K"),
                    "mda_raw": retriever.get_mda_raw("10-K"),
                    "balance_sheet_raw": retriever.extract_balance_sheet_as_str("tenk"),
                    "business_raw": retriever.get_business_raw(),
                    "cyber_raw": retriever.get_cybersecurity_raw(),
                    "legal_raw": retriever.get_legal_proceedings_raw(),
                    "market_risk_raw": retriever.get_market_risk_raw("10-K"),
                    "income_stmt_raw": retriever.get_income_statement("10-K"),
                    "cashflow_raw": retriever.get_cashflow_statement("10-K"),
                })
            result["tenk"] = tenk
    except (ValueError, Exception):
        # Foreign filers (BP, BABA, TSM) file 20-F instead of 10-K.
        try:
            retriever.get_twentyf_filing()
            meta = retriever._twentyf_metadata
            if meta:
                tenk = {
                    "metadata": {
                        **meta.to_dict(),
                        "edgar_url": _build_edgar_url(meta.cik, meta.accession),
                        "form_type": "20-F",
                    },
                }
                if fetch_raw:
                    tenk.update({
                        "risk_raw": retriever.get_risk_factors_raw("20-F"),
                        "mda_raw": retriever.get_mda_raw("20-F"),
                        "balance_sheet_raw": {},
                    })
                result["tenk"] = tenk
        except (ValueError, Exception):
            pass  # No 10-K or 20-F available

    # --- 10-Q ---
    # Foreign filers use 6-K, but edgartools' SixK class only wraps press
    # releases — no structured section access.
    try:
        retriever.get_tenq_filing()
        meta = retriever._tenq_metadata
        if meta:
            tenq: dict[str, Any] = {
                "metadata": {
                    **meta.to_dict(),
                    "edgar_url": _build_edgar_url(meta.cik, meta.accession),
                },
            }
            if fetch_raw:
                tenq.update({
                    "risk_raw": retriever.get_risk_factors_raw("10-Q"),
                    "mda_raw": retriever.get_mda_raw("10-Q"),
                    "income_stmt_raw": retriever.get_income_statement("10-Q"),
                    "cashflow_raw": retriever.get_cashflow_statement("10-Q"),
                })
            result["tenq"] = tenq
    except (ValueError, Exception):
        pass  # No 10-Q available

    # --- 8-K (earnings or material event) ---
    # The overview (kind + accession) is needed even keyless to pick the cache
    # key; only the per-item/earnings extraction is gated on fetch_raw.
    try:
        overview = retriever.get_8k_overview()
        if not overview.get("found"):
            result["eightk"] = {
                "kind": "none",
                "reason": overview.get("text", "No 8-K available"),
            }
        else:
            eightk_meta = retriever._eightk_metadata
            metadata = {
                **(eightk_meta.to_dict() if eightk_meta else {}),
                "edgar_url": _build_edgar_url(
                    eightk_meta.cik, eightk_meta.accession
                ) if eightk_meta else "",
                "form_type": "8-K",
            }
            kind = "earnings" if overview.get("has_earnings") else "event"
            if not fetch_raw:
                result["eightk"] = {"kind": kind, "raw": {}, "metadata": metadata}
            elif kind == "earnings":
                earnings = retriever.get_earnings_data()
                result["eightk"] = {
                    "kind": "earnings",
                    "raw": {**earnings, "metadata": metadata},
                    "metadata": metadata,
                }
            else:
                # Match the dict shape `_tool_material_event_summary` builds
                # so `analyze_material_event` accepts it directly.
                items = overview.get("items", [])
                primary_item = items[0] if items else None
                item_text = ""
                if primary_item:
                    item_result = retriever.get_8k_item(primary_item)
                    item_text = item_result.get("text", "") if item_result.get("found") else ""
                result["eightk"] = {
                    "kind": "event",
                    "raw": {
                        "content_type": overview.get("content_type", "other"),
                        "items": items,
                        "context": overview.get("context", ""),
                        "text": item_text,
                        "metadata": metadata,
                    },
                    "metadata": metadata,
                }
    except (ValueError, Exception) as e:
        result["eightk"] = {"kind": "none", "reason": f"No 8-K available ({e})"}

    return result


def _plan_sections(filing_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered list of analyzable sections from fetched filing data — the single
    source of truth for both endpoints (#10).

    Each entry: ``{section, key, form, accession, atype, input}``. ``section``/
    ``key`` place the result in the response; ``atype`` is the cache
    analysis_type (== key except 8-K, where atype is the 8-K kind). Conditional
    10-K sections (Item 1/1C/3/7A, XBRL) appear only when their raw text was
    extracted and found, so a keyless fetch (``fetch_raw=False`` → empty raw)
    yields just the always-present core sections.
    """
    plan: list[dict[str, Any]] = []

    if "tenk" in filing_data:
        tenk = filing_data["tenk"]
        acc = tenk["metadata"].get("accession", "")
        plan.append({"section": "tenk", "key": "risk_10k", "form": "10-K", "accession": acc, "atype": "risk_10k", "input": tenk.get("risk_raw", {})})
        plan.append({"section": "tenk", "key": "mda_10k", "form": "10-K", "accession": acc, "atype": "mda_10k", "input": tenk.get("mda_raw", {})})

        balance_input: dict[str, Any] = {"tenk": tenk.get("balance_sheet_raw", {})}
        if "tenq" in filing_data and filing_data["tenq"].get("balance_sheet_raw"):
            balance_input["tenq"] = filing_data["tenq"]["balance_sheet_raw"]
        plan.append({"section": "tenk", "key": "balance", "form": "10-K", "accession": acc, "atype": "balance", "input": balance_input})

        for atype, raw_key in [("business", "business_raw"), ("cybersecurity", "cyber_raw"), ("legal", "legal_raw"), ("market_risk", "market_risk_raw")]:
            raw = tenk.get(raw_key)
            if raw and raw.get("found"):
                plan.append({"section": "tenk", "key": atype, "form": "10-K", "accession": acc, "atype": atype, "input": raw})

        if tenk.get("income_stmt_raw"):
            income_input: dict[str, Any] = {"tenk": tenk["income_stmt_raw"], "tenk_metadata": tenk["metadata"]}
            if "tenq" in filing_data and filing_data["tenq"].get("income_stmt_raw"):
                income_input["tenq"] = filing_data["tenq"]["income_stmt_raw"]
                income_input["tenq_metadata"] = filing_data["tenq"]["metadata"]
            plan.append({"section": "tenk", "key": "income_stmt", "form": "10-K", "accession": acc, "atype": "income_stmt", "input": income_input})

        if tenk.get("cashflow_raw"):
            cashflow_input: dict[str, Any] = {"tenk": tenk["cashflow_raw"], "tenk_metadata": tenk["metadata"]}
            if "tenq" in filing_data and filing_data["tenq"].get("cashflow_raw"):
                cashflow_input["tenq"] = filing_data["tenq"]["cashflow_raw"]
                cashflow_input["tenq_metadata"] = filing_data["tenq"]["metadata"]
            plan.append({"section": "tenk", "key": "cashflow", "form": "10-K", "accession": acc, "atype": "cashflow", "input": cashflow_input})

    if "tenq" in filing_data:
        tenq = filing_data["tenq"]
        acc = tenq["metadata"].get("accession", "")
        plan.append({"section": "tenq", "key": "risk_10q", "form": "10-Q", "accession": acc, "atype": "risk_10q", "input": tenq.get("risk_raw", {})})
        plan.append({"section": "tenq", "key": "mda_10q", "form": "10-Q", "accession": acc, "atype": "mda_10q", "input": tenq.get("mda_raw", {})})

    eightk = filing_data.get("eightk", {})
    kind = eightk.get("kind", "none")
    if kind in ("earnings", "event"):
        acc = eightk.get("metadata", {}).get("accession", "")
        plan.append({"section": "eightk", "key": "analysis", "form": "8-K", "accession": acc, "atype": kind, "input": eightk.get("raw", {})})

    return plan


async def _analyze_or_cache(
    ticker: str, form_type: str, accession: str, analysis_type: str,
    llm_input: dict[str, Any], model_id: str, api_key: str | None,
    on_progress=None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve one filing section — the single source of truth for the
    cache → keyless-gate → generate flow shared by the JSON and SSE endpoints (#10).

    Returns ``(status, data)``: 'cached'|'needs_key'|'done'|'failed'. ``data``
    is None for needs_key/failed. ``on_progress(status, extra)`` (optional) is
    called as the state advances so the SSE endpoint can stream progress; the
    JSON endpoint passes None. The daily budget is charged by the caller's
    pre-flight (only when generation will occur), never here.
    """
    from api.db import get_filing_analysis, save_filing_analysis

    def progress(status: str, **extra):
        if on_progress:
            on_progress(status, extra)

    cached = await get_filing_analysis(ticker, form_type, accession, analysis_type)
    if cached:
        logger.info("[%s] CACHE HIT  %s/%s", ticker, form_type, analysis_type)
        progress("cached")
        return "cached", json.loads(cached["analysis_json"])

    if not api_key:
        # Cache miss with no key to generate — anonymous / keyless caller.
        progress("needs_key")
        return "needs_key", None

    progress("processing")
    logger.info("[%s] LLM CALL   %s/%s — calling %s", ticker, form_type, analysis_type, model_id)
    t0 = time.monotonic()
    try:
        async with llm_slot():
            analysis = await asyncio.to_thread(
                _run_llm_analysis, ticker, analysis_type, llm_input, model_id, api_key,
            )
        await save_filing_analysis(
            ticker, form_type, accession, analysis_type, json.dumps(analysis),
        )
        duration = round(time.monotonic() - t0, 1)
        logger.info("[%s] LLM DONE   %s/%s (%.1fs)", ticker, form_type, analysis_type, duration)
        progress("done", duration=duration)
        return "done", analysis
    except Exception as e:
        logger.error("[%s] LLM FAILED %s/%s: %s", ticker, form_type, analysis_type, e)
        progress("failed")
        return "failed", None


async def _charge_budget_if_generating(
    keys: ApiKeys, provider: str, ticker: str, plan: list[dict[str, Any]],
) -> bool:
    """Charge the operator's daily budget once, only when the request will
    actually run ≥1 LLM call (#6): operator-paid AND some planned section is a
    cache miss. A fully-cached (or keyless) request is never charged.

    Returns True if charging succeeded or wasn't needed; raises LLMBudgetExceeded
    when over budget so the caller can surface a 429 / SSE error.
    """
    if not keys.is_operator_paid(provider):
        return True
    from api.db import get_filing_analysis
    cached = await asyncio.gather(*[
        get_filing_analysis(ticker, p["form"], p["accession"], p["atype"]) for p in plan
    ])
    if any(c is None for c in cached):
        await check_and_charge_budget(keys.user_id)
    return True


def _run_llm_analysis(
    ticker: str, analysis_type: str, raw_data: dict[str, Any],
    model_id: str, api_key: str,
) -> dict[str, Any]:
    """Run a single LLM analysis and return the Pydantic model as a dict.

    analysis_type is one of: 'risk_10k', 'mda_10k', 'balance', 'risk_10q',
    'mda_10q', 'earnings', 'event', 'business', 'cybersecurity', 'legal',
    'market_risk', 'income_stmt', 'cashflow'.
    """
    from agents.llm_factory import create_llm
    from agents.sec_workflow.sec_llm_models import SECDocumentProcessor

    llm = create_llm(model_id, api_key)
    processor = SECDocumentProcessor(llm)

    if analysis_type == "risk_10k":
        result = processor.analyze_risk_factors(ticker, raw_data)
    elif analysis_type == "mda_10k":
        result = processor.analyze_mda(ticker, raw_data)
    elif analysis_type == "balance":
        # Balance sheet expects separate tenk/tenq dicts
        result = processor.analyze_balance_sheet(
            ticker, raw_data.get("tenk", {}), raw_data.get("tenq", {})
        )
    elif analysis_type == "risk_10q":
        result = processor.analyze_risk_factors(ticker, raw_data)
    elif analysis_type == "mda_10q":
        result = processor.analyze_mda(ticker, raw_data)
    elif analysis_type == "earnings":
        result = processor.analyze_earnings(ticker, raw_data)
    elif analysis_type == "event":
        result = processor.analyze_material_event(ticker, raw_data)
    elif analysis_type == "business":
        result = processor.analyze_business_overview(ticker, raw_data)
    elif analysis_type == "cybersecurity":
        result = processor.analyze_cybersecurity(ticker, raw_data)
    elif analysis_type == "legal":
        result = processor.analyze_legal_proceedings(ticker, raw_data)
    elif analysis_type == "market_risk":
        result = processor.analyze_market_risk(ticker, raw_data)
    elif analysis_type == "income_stmt":
        # Income statement combines tenk + tenq XBRL data
        result = processor.analyze_income_statement(ticker, raw_data)
    elif analysis_type == "cashflow":
        # Cash flow combines tenk + tenq XBRL data
        result = processor.analyze_cashflow(ticker, raw_data)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    return result.model_dump()


@router.get("/{ticker}/filings")
async def get_company_filings(
    ticker: str,
    request: Request,
    keys: ApiKeys = Depends(get_api_keys),
):
    """Return LLM-analyzed SEC filing summaries for the Filings tab.

    First call for a ticker runs LLM analysis (~5-15s, costs ~$0.005).
    Subsequent calls return instantly from SQLite (keyed by accession
    number — automatic cache invalidation when new filings are published).

    Cached analyses are public (no auth) for discoverability. Generating a
    fresh analysis on a cache miss needs an LLM API key — BYOK, or the
    operator key for signed-in users.
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    client_ip = request.client.host if request.client else "unknown"
    if not check_rest_rate_limit(
        rate_limit_key(keys.user_id, client_ip),
        bucket="filings",
        max_calls=10,
        window_seconds=3600,
    ):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")

    ticker = ticker.upper()

    # Resolve model + API key
    from agents.model_registry import get_model, get_default_model

    model_id = keys.model_id or get_default_model().id
    model = get_model(model_id) or get_default_model()
    # Key is optional: cached analyses are served to anyone. A key is only
    # needed to generate a fresh analysis on cache miss (BYOK, or the operator
    # env key for signed-in users). Keyless callers skip the expensive raw
    # section extraction — only metadata (the cache key) is fetched.
    api_key = keys.get_provider_key(model.provider)

    # Step 1: Fetch filing data from EDGAR (no LLM, ~1-3s). Raw section text is
    # extracted only when there's a key to generate with.
    logger.info("[%s] Fetching filings from EDGAR (raw=%s)...", ticker, bool(api_key))
    t0 = time.monotonic()
    filing_data = await asyncio.to_thread(_fetch_filing_data, ticker, api_key is not None)
    logger.info("[%s] EDGAR fetch complete (%.1fs)", ticker, time.monotonic() - t0)

    plan = _plan_sections(filing_data)

    # Charge the daily budget once, and only when an LLM call will actually run
    # (operator-paid + ≥1 cache miss). Fully-cached / keyless reads cost nothing.
    try:
        await _charge_budget_if_generating(keys, model.provider, ticker, plan)
    except LLMBudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    # Resolve every section concurrently — cache hits return instantly, misses
    # with a key run the LLM in parallel threads.
    results = await asyncio.gather(*[
        _analyze_or_cache(ticker, p["form"], p["accession"], p["atype"], p["input"], model.id, api_key)
        for p in plan
    ])

    response: dict[str, Any] = {"ticker": ticker}
    if "tenk" in filing_data:
        response["tenk"] = {"metadata": filing_data["tenk"]["metadata"]}
    if "tenq" in filing_data:
        response["tenq"] = {"metadata": filing_data["tenq"]["metadata"]}

    eightk = filing_data.get("eightk", {})
    eightk_kind = eightk.get("kind", "none")

    for p, (_status, data) in zip(plan, results):
        if data is None:
            continue
        if p["section"] in ("tenk", "tenq"):
            response[p["section"]][p["key"]] = data
        elif p["section"] == "eightk":
            response["eightk"] = {
                "kind": eightk_kind,
                "metadata": eightk.get("metadata", {}),
                "analysis": data,
            }

    if "eightk" not in response:
        response["eightk"] = {
            "kind": eightk_kind,
            "reason": eightk.get("reason", "No 8-K data available"),
        }

    return JSONResponse(content=response)


@router.get("/{ticker}/filings/stream")
async def stream_company_filings(
    ticker: str,
    request: Request,
    keys: ApiKeys = Depends(get_api_keys),
):
    """Stream SEC filing analysis as Server-Sent Events.

    Emits progress events as each LLM call starts/completes, then a
    ``section`` event with the full analysis payload.  All LLM analyses
    run concurrently, so wall-clock time equals the slowest single call
    rather than the sum of all calls.

    Event shapes::

        {"type": "progress", "step": "edgar_fetch",       "status": "fetching"}
        {"type": "progress", "step": "edgar_fetch",       "status": "complete", "duration": 8.3}
        {"type": "metadata", "tenk_metadata": {...},      "tenq_metadata": {...}, ...}
        {"type": "progress", "step": "10-K/risk_10k",     "status": "processing"}
        {"type": "progress", "step": "10-K/risk_10k",     "status": "done",  "duration": 14.1}
        {"type": "progress", "step": "10-K/risk_10k",     "status": "cached"}
        {"type": "section",  "form": "10-K", "key": "risk_10k", "data": {...}}
        {"type": "complete"}
        {"type": "error",    "message": "..."}
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ticker = ticker.upper()

    from agents.model_registry import get_model, get_default_model

    model_id = keys.model_id or get_default_model().id
    model = get_model(model_id) or get_default_model()
    # Key is optional: cached sections stream to anyone; a key is only needed
    # to generate fresh sections on cache miss (BYOK, or operator key signed-in).
    # Keyless callers skip raw section extraction (only metadata is fetched).
    api_key = keys.get_provider_key(model.provider)

    # Per-IP concurrency guard — prevents a single client from opening
    # many concurrent streams and exhausting the thread pool or API credits.
    client_ip = request.client.host if request.client else "unknown"
    semaphore = _get_stream_semaphore(client_ip)
    if semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent filing streams (max {_MAX_CONCURRENT_STREAMS})",
        )

    # Hourly volume cap, complementary to the concurrency cap above.
    if not check_rest_rate_limit(
        rate_limit_key(keys.user_id, client_ip),
        bucket="filings_stream",
        max_calls=10,
        window_seconds=3600,
    ):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")

    async def generate():
        # Acquire a concurrency slot for the duration of this stream.
        # Released when the generator finishes (normal exit or client disconnect).
        await semaphore.acquire()
        try:
            # Step 1: Fetch filings from EDGAR. Raw section text is extracted
            # only when there's a key to generate with (keyless → metadata only).
            yield _sse({"type": "progress", "step": "edgar_fetch", "status": "fetching"})
            try:
                t0 = time.monotonic()
                filing_data = await asyncio.to_thread(_fetch_filing_data, ticker, api_key is not None)
                logger.info("[%s] EDGAR fetch complete (%.1fs)", ticker, time.monotonic() - t0)
                yield _sse({
                    "type": "progress",
                    "step": "edgar_fetch",
                    "status": "complete",
                    "duration": round(time.monotonic() - t0, 1),
                })
            except Exception as e:
                logger.error("[%s] EDGAR fetch failed: %s", ticker, e)
                yield _sse({"type": "error", "message": "Failed to fetch filings from EDGAR."})
                return

            # Emit metadata immediately so the frontend can render section headers
            eightk = filing_data.get("eightk", {})
            eightk_kind = eightk.get("kind", "none")
            yield _sse({
                "type": "metadata",
                "tenk_metadata": filing_data.get("tenk", {}).get("metadata"),
                "tenq_metadata": filing_data.get("tenq", {}).get("metadata"),
                "eightk_kind": eightk_kind,
                "eightk_metadata": eightk.get("metadata") if eightk_kind != "none" else None,
            })

            plan = _plan_sections(filing_data)

            # Charge the daily budget once, only when an LLM call will run
            # (operator-paid + ≥1 cache miss). Over budget → error + stop.
            try:
                await _charge_budget_if_generating(keys, model.provider, ticker, plan)
            except LLMBudgetExceeded as e:
                yield _sse({"type": "error", "message": str(e)})
                return

            # Run all sections concurrently using a shared queue. Each task pushes
            # progress + section events as they happen; the outer generator yields
            # them in real time, so wall-clock is the *max* of the LLM calls, not
            # the sum. _analyze_or_cache is the same resolver the JSON endpoint uses.
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def section_task(p: dict[str, Any]):
                step = f"{p['form']}/{p['atype']}"

                def on_progress(status: str, extra: dict):
                    queue.put_nowait(_sse({"type": "progress", "step": step, "status": status, **extra}))

                _status, data = await _analyze_or_cache(
                    ticker, p["form"], p["accession"], p["atype"], p["input"],
                    model.id, api_key, on_progress=on_progress,
                )
                if data is not None:
                    await queue.put(_sse({"type": "section", "form": p["form"], "key": p["key"], "data": data}))

            tasks = [section_task(p) for p in plan]

            if tasks:
                async def run_all():
                    """Gather all section tasks, then send sentinel to signal completion."""
                    await asyncio.gather(*tasks)
                    await queue.put(None)  # sentinel

                runner = asyncio.create_task(run_all())

                # Yield events as they arrive from the concurrent tasks
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield event

                # Propagate any unexpected errors from the gather
                await runner

            yield _sse({"type": "complete"})
        finally:
            semaphore.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
