"""Company dashboard endpoints — profile and filing analysis.

Profile endpoint aggregates yfinance, technicals, patterns, and market regime
into a single JSON response with no LLM calls (< 1s).

Filings endpoint runs LLM analysis on SEC filings with DB-first caching:
first load is slow (LLM calls), every subsequent load is instant (SQLite).
Analyses are keyed by accession number; which accession is "latest" comes
from the shared freshness check in ``agents/sec_workflow/analysis_cache.py``
(the same one the chat SEC tools use), which asks EDGAR at most once per
``FILING_FRESHNESS_TTL_SECONDS``.
"""

import asyncio
import json
import logging
import time
from typing import Any

from cachetools import TTLCache
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agents.sec_workflow.analysis_cache import (
    FORMS,
    FilingRef,
    eightk_overview_and_input,
    eightk_kind,
    get_or_analyze,
    get_retriever,
    latest_filings,
    run_analysis,
)
from agents.sec_workflow.get_SEC_data import _load_report
from api.dependencies import ApiKeys, get_api_keys, resolve_model
from api.llm_concurrency import (
    LLMBudgetExceeded,
    LLMCallCharger,
    llm_slot,
    read_budget_usage,
    refusal_message,
)
from api.rate_limit import check_rest_rate_limit, client_ip, rate_limit_key
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


def _get_stream_semaphore(ip: str) -> asyncio.Semaphore:
    """Return the per-IP stream semaphore, creating one on first sight.

    Wraps the TTLCache lookup so callers don't repeat the get-or-create dance.
    """
    sem = _stream_semaphores.get(ip)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)
        _stream_semaphores[ip] = sem
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
    # Timestamps, etc. to JSON-safe primitives. allow_nan=False matches
    # JSONResponse, so an unrenderable payload fails here — before it's cached.
    return await asyncio.to_thread(lambda: json.loads(json.dumps(raw, default=str, allow_nan=False)))


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


def _paired_read(retriever, form: str, read) -> tuple[Any, dict[str, Any]]:
    """``(read(), metadata of the filing it read)`` for a getter that doesn't
    report its filing (the XBRL statements). The shared retriever can move to
    a newer filing mid-request (a concurrent freshness refresh), so the
    filing (report + metadata, read together) is checked before and after;
    a read that straddled a swap is retried once, then dropped as
    ``(None, {})``."""
    try:
        for _ in range(2):
            report, meta = _load_report(retriever, form)
            value = read()
            after_report, after_meta = _load_report(retriever, form)
            if after_report is report and after_meta is meta:
                return value, meta.to_dict() if meta else {}
        logger.info("[%s] %s changed during every read; section skipped", retriever.ticker, form)
    except Exception as e:
        logger.warning("[%s] %s read failed: %s", retriever.ticker, form, e)
    return None, {}


def _balance_sheet_read(retriever, form: str) -> tuple[str, dict[str, Any]]:
    """``(balance sheet text, metadata of its filing)``; ``""`` when it
    can't be extracted (the section is then planned cache_only)."""
    try:
        text, meta = retriever.balance_sheet_text(form)
    except Exception as e:
        logger.warning("[%s] %s balance sheet extraction failed: %s", retriever.ticker, form, e)
        return "", {}
    return text, meta.to_dict() if meta else {}


def _statement_reads(retriever, form: str) -> dict[str, Any]:
    """XBRL income and cash-flow statements of ``form``, each with the
    metadata of the filing it was read from (``read_metadata``)."""
    income, income_meta = _paired_read(retriever, form, lambda: retriever.get_income_statement(form))
    cashflow, cashflow_meta = _paired_read(retriever, form, lambda: retriever.get_cashflow_statement(form))
    return {
        "income_stmt_raw": income,
        "cashflow_raw": cashflow,
        "read_metadata": {"income_stmt_raw": income_meta, "cashflow_raw": cashflow_meta},
    }


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

    Reads the shared per-ticker retriever; ``_load_filing_data`` syncs it with
    EDGAR's latest filings first. A concurrent refresh can still move it to a
    newer filing mid-extraction, so every raw read carries the metadata of
    the filing it actually came from (section results their ``metadata``,
    the rest ``read_metadata``) and ``_plan_sections`` keys by that.
    """
    retriever = get_retriever(ticker)
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
                tenk["risk_raw"] = retriever.get_risk_factors_raw("10-K")
                tenk["mda_raw"] = retriever.get_mda_raw("10-K")
                tenk["balance_sheet_raw"], balance_meta = _balance_sheet_read(retriever, "10-K")
                tenk.update({
                    "business_raw": retriever.get_business_raw(),
                    "cyber_raw": retriever.get_cybersecurity_raw(),
                    "legal_raw": retriever.get_legal_proceedings_raw(),
                    "market_risk_raw": retriever.get_market_risk_raw("10-K"),
                    **_statement_reads(retriever, "10-K"),
                })
                tenk["read_metadata"]["balance_sheet_raw"] = balance_meta
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
                    **_statement_reads(retriever, "10-Q"),
                })
            result["tenq"] = tenq
    except (ValueError, Exception):
        pass  # No 10-Q available

    # --- 8-K (earnings or material event) ---
    # The overview (kind + accession) is needed even keyless to pick the cache
    # key; only the per-item/earnings extraction is gated on fetch_raw. The
    # overview, its metadata (the cache key) and the input all come from one
    # loaded 8-K, as in the chat 8-K tool.
    try:
        if fetch_raw:
            overview, raw = eightk_overview_and_input(retriever)
        else:
            overview, raw = retriever.get_8k_overview(), {}
        if not overview.get("found"):
            result["eightk"] = {
                "kind": "none",
                "reason": overview.get("text", "No 8-K available"),
            }
        else:
            read_meta = overview.get("metadata") or {}
            metadata = {
                **read_meta,
                "edgar_url": _build_edgar_url(read_meta.get("cik", ""), read_meta.get("accession", "")),
                "form_type": "8-K",
            }
            kind = eightk_kind(overview)
            result["eightk"] = {"kind": kind, "raw": raw, "metadata": metadata}
            if fetch_raw and kind == "event" and not raw["text"]:
                primary_item = raw["items"][0] if raw["items"] else None
                result["eightk"]["reason"] = f"8-K item {primary_item} could not be extracted"
    except (ValueError, Exception) as e:
        result["eightk"] = {"kind": "none", "reason": f"No 8-K available ({e})"}

    return result


async def _load_filing_data(ticker: str, fetch_raw: bool) -> dict[str, Any]:
    """Point the shared retriever at EDGAR's latest 10-K, 20-F, 10-Q and 8-K
    (the freshness check chat uses too — TTL-cached, so usually no SEC
    request, and one submissions request for all forms when it does ask),
    then run ``_fetch_filing_data`` off the event loop. A failed lookup is
    logged and the retriever's current filings are served."""
    try:
        await latest_filings(ticker, FORMS)
    except Exception as e:
        logger.warning("[%s] latest-filing lookup failed: %s", ticker, e)
    return await asyncio.to_thread(_fetch_filing_data, ticker, fetch_raw)


def _filing_ref(p: dict[str, Any]) -> FilingRef:
    """The cache key filing for a planned section."""
    return FilingRef(p["form"], p["accession"], p["filing_date"])


def _keyed_by(form_meta: dict[str, Any], read_meta: dict[str, Any] | None) -> dict[str, str]:
    """``{accession, filing_date}`` a section is keyed by: the filing its input
    was read from, else (nothing read — keyless, or a failed read) the
    form's metadata."""
    meta = read_meta if read_meta and read_meta.get("accession") else form_meta
    return {"accession": meta.get("accession", ""), "filing_date": meta.get("filing_date", "")}


def _plan_sections(filing_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered list of analyzable sections from fetched filing data — the single
    source of truth for both endpoints (#10).

    Each entry: ``{section, key, form, accession, filing_date, atype, input,
    cache_only}``. ``accession``/``filing_date`` (the cache key) are those of
    the filing the section's input was read from (``_keyed_by``).
    ``section``/``key`` place the result in the response; ``atype`` is the cache
    analysis_type (== key except 8-K, where atype is the 8-K kind). Conditional
    10-K sections (Item 1/1C/3/7A, XBRL) appear only when their raw text was
    extracted and found. A keyless fetch (``fetch_raw=False`` → no raw) plans
    the always-present core sections normally and those conditional 10-K
    sections ``cache_only`` (the filing may not contain them, so a miss must
    not surface ``needs_key``).

    Core sections whose raw text *was* fetched but failed to extract
    (``found: False``, an empty 20-F balance sheet, an 8-K without parseable
    earnings, an event 8-K whose item text is missing) are planned
    ``cache_only``: an existing cached analysis is still served, but a cache
    miss never generates — otherwise the LLM analyses the error message or
    bare metadata and the result is cached for the filing's lifetime.
    Unfetched (keyless) input is ``None``/empty and is planned normally.
    """
    plan: list[dict[str, Any]] = []

    def failed(raw: dict[str, Any] | None) -> bool:
        return raw is not None and not raw.get("found")

    def section_meta(raw: dict[str, Any] | None) -> dict[str, Any] | None:
        return raw.get("metadata") if raw else None

    def read_meta(form_data: dict[str, Any], raw_key: str) -> dict[str, Any]:
        """Metadata of the filing ``raw_key`` was read from, else the form's."""
        meta = form_data.get("read_metadata", {}).get(raw_key)
        return meta if meta and meta.get("accession") else form_data["metadata"]

    if "tenk" in filing_data:
        tenk = filing_data["tenk"]
        form_meta = tenk["metadata"]
        risk_raw, mda_raw = tenk.get("risk_raw"), tenk.get("mda_raw")
        plan.append({"section": "tenk", "key": "risk_10k", "form": "10-K", **_keyed_by(form_meta, section_meta(risk_raw)), "atype": "risk_10k", "input": risk_raw or {}, "cache_only": failed(risk_raw)})
        plan.append({"section": "tenk", "key": "mda_10k", "form": "10-K", **_keyed_by(form_meta, section_meta(mda_raw)), "atype": "mda_10k", "input": mda_raw or {}, "cache_only": failed(mda_raw)})

        balance_raw = tenk.get("balance_sheet_raw")
        balance_input: dict[str, Any] = {"tenk": balance_raw or {}}
        if "tenq" in filing_data and filing_data["tenq"].get("balance_sheet_raw"):
            balance_input["tenq"] = filing_data["tenq"]["balance_sheet_raw"]
        plan.append({"section": "tenk", "key": "balance", "form": "10-K", **_keyed_by(form_meta, read_meta(tenk, "balance_sheet_raw")), "atype": "balance", "input": balance_input, "cache_only": balance_raw is not None and not balance_raw})

        # Conditional 10-K sections. A keyless fetch extracted no raw text, so
        # there is nothing to learn about whether the filing has them: plan each
        # as a cache-only lookup (served if a keyed caller already analysed it,
        # silently omitted otherwise — never a needs_key placeholder).
        # 20-F filers have none of these sections.
        keyless_10k = "risk_raw" not in tenk and form_meta.get("form_type") == "10-K"

        for atype, raw_key in [("business", "business_raw"), ("cybersecurity", "cyber_raw"), ("legal", "legal_raw"), ("market_risk", "market_risk_raw")]:
            raw = tenk.get(raw_key)
            if raw and raw.get("found"):
                plan.append({"section": "tenk", "key": atype, "form": "10-K", **_keyed_by(form_meta, section_meta(raw)), "atype": atype, "input": raw, "cache_only": False})
            elif keyless_10k:
                plan.append({"section": "tenk", "key": atype, "form": "10-K", **_keyed_by(form_meta, None), "atype": atype, "input": {}, "cache_only": True})

        for atype, raw_key in [("income_stmt", "income_stmt_raw"), ("cashflow", "cashflow_raw")]:
            if not tenk.get(raw_key):
                if keyless_10k:
                    plan.append({"section": "tenk", "key": atype, "form": "10-K", **_keyed_by(form_meta, None), "atype": atype, "input": {}, "cache_only": True})
                continue
            tenk_meta = read_meta(tenk, raw_key)
            statement_input: dict[str, Any] = {"tenk": tenk[raw_key], "tenk_metadata": tenk_meta}
            if "tenq" in filing_data and filing_data["tenq"].get(raw_key):
                statement_input["tenq"] = filing_data["tenq"][raw_key]
                statement_input["tenq_metadata"] = read_meta(filing_data["tenq"], raw_key)
            plan.append({"section": "tenk", "key": atype, "form": "10-K", **_keyed_by(form_meta, tenk_meta), "atype": atype, "input": statement_input, "cache_only": False})

    if "tenq" in filing_data:
        tenq = filing_data["tenq"]
        form_meta = tenq["metadata"]
        risk_raw, mda_raw = tenq.get("risk_raw"), tenq.get("mda_raw")
        plan.append({"section": "tenq", "key": "risk_10q", "form": "10-Q", **_keyed_by(form_meta, section_meta(risk_raw)), "atype": "risk_10q", "input": risk_raw or {}, "cache_only": failed(risk_raw)})
        plan.append({"section": "tenq", "key": "mda_10q", "form": "10-Q", **_keyed_by(form_meta, section_meta(mda_raw)), "atype": "mda_10q", "input": mda_raw or {}, "cache_only": failed(mda_raw)})

    eightk = filing_data.get("eightk", {})
    kind = eightk.get("kind", "none")
    if kind in ("earnings", "event"):
        raw = eightk.get("raw", {})
        # Keyless fetches leave raw empty; only a fetched-but-unusable 8-K is cache_only.
        if kind == "earnings":
            unusable = raw.get("has_earnings") is False
        else:
            unusable = bool(raw) and not raw.get("text")
        plan.append({"section": "eightk", "key": "analysis", "form": "8-K", **_keyed_by(eightk.get("metadata", {}), None), "atype": kind, "input": raw, "cache_only": unusable})

    return plan


async def _reported_eightk_kind(
    ticker: str, filing_data: dict[str, Any], plan: list[dict[str, Any]],
) -> str:
    """The 8-K kind both endpoints report — single source of truth so the JSON
    and SSE responses cannot drift.

    An 8-K planned ``cache_only`` (item text failed to extract) is reported as
    ``"none"`` unless a cached analysis exists for it; otherwise clients render
    an empty 8-K heading that no section will ever fill.
    """
    kind = filing_data.get("eightk", {}).get("kind", "none")
    entry = next((p for p in plan if p["section"] == "eightk"), None)
    if entry is not None and entry["cache_only"]:
        from api.db import get_filing_analysis
        cached = await get_filing_analysis(
            ticker, entry["form"], entry["accession"], entry["atype"],
        )
        if not cached:
            return "none"
    return kind


async def _analyze_or_cache(
    ticker: str, filing: FilingRef, analysis_type: str,
    llm_input: dict[str, Any], model_id: str, api_key: str | None,
    on_progress=None, cache_only: bool = False, charger: LLMCallCharger | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve one filing section — the single source of truth for the
    cache → keyless-gate → generate flow shared by the JSON and SSE endpoints (#10).
    Cache read/generate/save is the shared ``get_or_analyze`` the chat tools use.

    Returns ``(status, data)``:
    'cached'|'needs_key'|'unavailable'|'done'|'failed'|'budget_exceeded'.
    ``data`` (the analysis plus its ``filing_reference``) is None for everything
    but cached/done. ``cache_only`` (input failed to extract) serves a cache
    hit but returns 'unavailable' on a miss, silently.
    ``on_progress(status, extra)`` (optional) is called as the state advances so the SSE endpoint can stream progress; the
    JSON endpoint passes one only for the 8-K, to read its refusal message.
    ``charger`` (operator-paid requests only, from
    `_budget_charger`) charges the daily budget as this section's LLM call
    starts and refuses the call the budget can't cover ('budget_exceeded',
    with the user-facing `refusal_message` as the progress ``message``);
    cache hits and skipped sections cost nothing.
    """
    form_type = filing.form

    def progress(status: str, **extra):
        if on_progress:
            on_progress(status, extra)

    started: list[float] = []

    async def generate() -> dict[str, Any]:
        from agents.llm_factory import create_llm

        progress("processing")
        logger.info("[%s] LLM CALL   %s/%s — calling %s", ticker, form_type, analysis_type, model_id)
        started.append(time.monotonic())
        async with llm_slot():
            llm = await asyncio.to_thread(create_llm, model_id, api_key)
            if charger is not None:
                # Charged as the LLM call starts, inside the slot: a saturated
                # pool fails the section before any call, so it costs nothing.
                llm = llm.with_config(callbacks=[charger])
            return await run_analysis(llm, ticker, analysis_type, llm_input)

    try:
        result = await get_or_analyze(
            ticker, filing, analysis_type,
            None if cache_only or not api_key else generate,
        )
    except LLMBudgetExceeded as e:
        logger.info("[%s] BUDGET     %s/%s refused: %s", ticker, form_type, analysis_type, e)
        progress("budget_exceeded", message=refusal_message(e))
        return "budget_exceeded", None
    except Exception as e:
        logger.error("[%s] LLM FAILED %s/%s: %s", ticker, form_type, analysis_type, e)
        progress("failed")
        return "failed", None

    if result is None:
        if cache_only:
            return "unavailable", None
        # Cache miss with no key to generate — anonymous / keyless caller.
        progress("needs_key")
        return "needs_key", None

    if not result.generated:
        logger.info("[%s] CACHE HIT  %s/%s", ticker, form_type, analysis_type)
        progress("cached")
        return "cached", result.served()

    duration = round(time.monotonic() - started[0], 1)
    logger.info("[%s] LLM DONE   %s/%s (%.1fs)", ticker, form_type, analysis_type, duration)
    progress("done", duration=duration)
    return "done", result.served()


def _budget_user(keys: ApiKeys, api_key: str | None, provider: str) -> str | None:
    """The user to charge per LLM analysis, or None when the operator isn't
    paying (BYOK, or no key that may run the model)."""
    if api_key is not None and keys.is_operator_paid(provider):
        return keys.user_id
    return None


async def _budget_charger(charge_user_id: str | None) -> LLMCallCharger | None:
    """The per-call budget charger for an operator-paid request (None when
    the operator isn't paying). Must run on the event loop.

    No request-level refusal: the charger charges each section's LLM call as
    it starts and refuses the calls the budget can't cover
    (`_analyze_or_cache` → 'budget_exceeded'), so with today's budget spent
    cached sections are still served and only the uncached ones are refused.
    Cache hits, BYOK and keyless requests are never charged.
    """
    if charge_user_id is None:
        return None
    return LLMCallCharger(charge_user_id, used=await read_budget_usage(charge_user_id))


@router.get("/{ticker}/filings")
async def get_company_filings(
    ticker: str,
    request: Request,
    keys: ApiKeys = Depends(get_api_keys),
    x_model_id: str | None = Header(None),
):
    """Return LLM-analyzed SEC filing summaries for the Filings tab.

    First call for a ticker runs LLM analysis (~5-15s; up to 12 calls, ~70-100k
    input tokens — about $0.07 per company on gemini-3.1-flash-lite).
    Subsequent calls return instantly from SQLite (keyed by accession
    number; a newly published filing is picked up within
    FILING_FRESHNESS_TTL_SECONDS and analysed on its first request).

    Cached analyses are public (no auth) for discoverability. Generating a
    fresh analysis on a cache miss needs an LLM API key — BYOK, or the
    operator key for signed-in users.
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ip = client_ip(request)
    if not check_rest_rate_limit(
        rate_limit_key(keys.user_id, ip),
        bucket="filings",
        max_calls=10,
        window_seconds=3600,
    ):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")

    ticker = ticker.upper()

    # Resolve model + API key
    # Key is optional: cached analyses are served to anyone. A key is only
    # needed to generate a fresh analysis on cache miss (BYOK, or the operator
    # env key for signed-in users on a model it may lend — otherwise the
    # caller is keyless here and misses report needs_key). Keyless callers
    # skip the expensive raw section extraction — only metadata (the cache
    # key) is fetched. Absent an explicit X-Model-Id, the model is one the
    # caller has a usable key for, not necessarily the registry default.
    model, api_key = resolve_model(keys, x_model_id)

    # Step 1: Fetch filing data from EDGAR (no LLM, ~1-3s). Raw section text is
    # extracted only when there's a key to generate with.
    logger.info("[%s] Fetching filings from EDGAR (raw=%s)...", ticker, bool(api_key))
    t0 = time.monotonic()
    filing_data = await _load_filing_data(ticker, api_key is not None)
    logger.info("[%s] EDGAR fetch complete (%.1fs)", ticker, time.monotonic() - t0)

    plan = _plan_sections(filing_data)

    # Operator-paid: each generated section is charged as its LLM call starts;
    # sections the daily budget can't cover are left out and listed in
    # `budget_exceeded`. Cache hits / BYOK / keyless reads cost nothing.
    charger = await _budget_charger(_budget_user(keys, api_key, model.provider))

    # The refusal message a refused 8-K reports as its `reason`.
    eightk_refusal: list[str] = []

    def on_eightk_progress(status: str, extra: dict[str, Any]) -> None:
        if status == "budget_exceeded":
            eightk_refusal.append(extra["message"])

    # Resolve every section concurrently — cache hits return instantly, misses
    # with a key run their LLM calls concurrently.
    results = await asyncio.gather(*[
        _analyze_or_cache(
            ticker, _filing_ref(p), p["atype"], p["input"], model.id, api_key,
            on_progress=on_eightk_progress if p["section"] == "eightk" else None,
            cache_only=p["cache_only"], charger=charger,
        )
        for p in plan
    ])
    if charger is not None:
        await charger.settle()

    # Steps (same "{form}/{atype}" names as the SSE progress events) whose
    # analysis was refused because the daily budget is spent.
    response: dict[str, Any] = {
        "ticker": ticker,
        "budget_exceeded": [
            f"{p['form']}/{p['atype']}"
            for p, (status, _data) in zip(plan, results) if status == "budget_exceeded"
        ],
    }
    if "tenk" in filing_data:
        response["tenk"] = {"metadata": filing_data["tenk"]["metadata"]}
    if "tenq" in filing_data:
        response["tenq"] = {"metadata": filing_data["tenq"]["metadata"]}

    eightk = filing_data.get("eightk", {})
    eightk_kind = await _reported_eightk_kind(ticker, filing_data, plan)

    for p, (status, data) in zip(plan, results):
        if p["section"] == "eightk" and status == "budget_exceeded":
            response["eightk"] = {
                "kind": eightk_kind,
                "metadata": eightk.get("metadata", {}),
                "reason": eightk_refusal[0],
            }
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
    x_model_id: str | None = Header(None),
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
        {"type": "progress", "step": "10-K/risk_10k",     "status": "budget_exceeded", "message": "..."}
        {"type": "section",  "form": "10-K", "key": "risk_10k", "data": {...}}
        {"type": "complete"}
        {"type": "error",    "message": "..."}
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ticker = ticker.upper()

    # Key is optional: cached sections stream to anyone; a key is only needed
    # to generate fresh sections on cache miss (BYOK, or operator key signed-in,
    # for models it may lend). Keyless callers skip raw section extraction
    # (only metadata is fetched).
    model, api_key = resolve_model(keys, x_model_id)
    charge_user_id = _budget_user(keys, api_key, model.provider)

    # Per-IP concurrency guard — prevents a single client from opening
    # many concurrent streams and exhausting the thread pool or API credits.
    ip = client_ip(request)
    semaphore = _get_stream_semaphore(ip)
    if semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent filing streams (max {_MAX_CONCURRENT_STREAMS})",
        )

    # Hourly volume cap, complementary to the concurrency cap above.
    if not check_rest_rate_limit(
        rate_limit_key(keys.user_id, ip),
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
                filing_data = await _load_filing_data(ticker, api_key is not None)
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

            # Plan before metadata: the reported 8-K kind depends on whether an
            # unusable 8-K has a cached analysis to show.
            plan = _plan_sections(filing_data)

            # Emit metadata immediately so the frontend can render section headers
            eightk = filing_data.get("eightk", {})
            eightk_kind = await _reported_eightk_kind(ticker, filing_data, plan)
            yield _sse({
                "type": "metadata",
                "tenk_metadata": filing_data.get("tenk", {}).get("metadata"),
                "tenq_metadata": filing_data.get("tenq", {}).get("metadata"),
                "eightk_kind": eightk_kind,
                "eightk_metadata": eightk.get("metadata") if eightk_kind != "none" else None,
            })

            # Operator-paid: each generated section is charged as its LLM call
            # starts; sections the budget can't cover report 'budget_exceeded'
            # while cached ones are still served.
            charger = await _budget_charger(charge_user_id)

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
                    ticker, _filing_ref(p), p["atype"], p["input"],
                    model.id, api_key, on_progress=on_progress, cache_only=p["cache_only"],
                    charger=charger,
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

            if charger is not None:
                await charger.settle()
            yield _sse({"type": "complete"})
        finally:
            semaphore.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
