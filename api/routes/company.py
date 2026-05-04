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

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.dependencies import ApiKeys, get_api_keys
from api.validators import TICKER_RE

logger = logging.getLogger("analyst.filings")

router = APIRouter(prefix="/api/company")

# Server-side cache: {ticker: (monotonic_timestamp, payload)}
# Values are JSON-round-tripped dicts to guarantee serializability.
_profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_profile_locks: dict[str, asyncio.Lock] = {}
_PROFILE_CACHE_TTL = 300  # 5 minutes

# Per-IP concurrency limiter for SSE streams.
# Each stream holds up to 6 LLM calls — capping concurrent streams per IP
# prevents a single client from exhausting the thread pool or burning API credits.
_MAX_CONCURRENT_STREAMS = 2
_stream_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)
)


def _build_profile(ticker: str) -> dict[str, Any]:
    """Synchronous profile assembly — runs in a thread pool.

    Aggregates four independent data sources:
    1. yfinance company info + live quote + earnings calendar
    2. Technical indicators (RSI, MACD, ADX, etc.)
    3. Chart pattern detection
    4. Market regime (SPY/VIX-based)

    Each source fails independently — partial data is returned rather
    than failing the whole request.
    """
    from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
    from agents.technical_workflow.process_technical_indicators import TechnicalIndicators
    from agents.technical_workflow.pattern_recognition import PatternRecognitionEngine
    from agents.market_analysis.regime_detector import MarketRegimeDetector

    retriever = YahooFinanceDataRetrieval(ticker)

    # --- Company info + quote + earnings ---
    company = retriever.get_company_profile()
    quote = retriever.get_live_price()
    earnings = retriever.get_earnings_calendar()

    # --- Technical indicators (needs 1y daily OHLCV) ---
    technicals: dict[str, Any] = {}
    patterns: list[dict[str, Any]] = []
    df = retriever.get_historical_prices(period="1y", interval="1d")
    if df is not None and not df.empty:
        ti = TechnicalIndicators(ticker)
        all_ind = ti.calculate_all_indicators(df)

        # Extract the dashboard-relevant snapshot from the full indicator set
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
            raw = engine.detect_all_patterns(df)
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
            pass  # Patterns are supplementary

    # --- Market regime ---
    regime: dict[str, Any] = {}
    try:
        regime = MarketRegimeDetector().detect_regime()
    except Exception:
        pass  # Regime is supplementary

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
    return json.loads(json.dumps(raw, default=str))


@router.get("/{ticker}/profile")
async def get_company_profile(ticker: str):
    """Return aggregated company profile for the dashboard Overview tab.

    Combines yfinance metadata, live quote, technical indicators, chart
    patterns, and market regime into a single response. No LLM calls —
    pure data aggregation, typically responds in < 1 second.
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ticker = ticker.upper()

    # Per-ticker lock prevents duplicate in-flight fetches for the same ticker.
    lock = _profile_locks.setdefault(ticker, asyncio.Lock())
    async with lock:
        # Check server-side cache (inside lock to avoid thundering herd)
        now = time.monotonic()
        cached = _profile_cache.get(ticker)
        if cached and (now - cached[0]) < _PROFILE_CACHE_TTL:
            return JSONResponse(
                content=cached[1],
                headers={"Cache-Control": f"public, max-age={_PROFILE_CACHE_TTL}"},
            )

        # Run synchronous data fetching in a thread to avoid blocking the event loop.
        # _build_profile handles JSON round-trip internally for numpy/Timestamp safety.
        payload = await asyncio.to_thread(_build_profile, ticker)
        _profile_cache[ticker] = (time.monotonic(), payload)

        # Track the company as soon as its profile is first loaded.
        # This is the earliest signal that a user is interested in a ticker —
        # before any chat session or filings analysis is triggered.
        from api.db import ensure_company
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


async def _stream_section(
    form_type: str,
    accession: str,
    analysis_type: str,
    raw_data: dict[str, Any],
    ticker: str,
    model_id: str,
    api_key: str,
):
    """Async generator for one filing analysis section.

    Yields SSE event strings for progress updates, then yields a
    ``("result", analysis_or_None)`` sentinel as the final item so the
    caller can capture the data without breaking the yield chain.
    """
    from api.db import get_filing_analysis, save_filing_analysis

    step = f"{form_type}/{analysis_type}"
    cached = await get_filing_analysis(ticker, form_type, accession, analysis_type)
    if cached:
        logger.info("[%s] CACHE HIT  %s/%s", ticker, form_type, analysis_type)
        yield _sse({"type": "progress", "step": step, "status": "cached"})
        yield ("result", json.loads(cached["analysis_json"]))
        return

    logger.info("[%s] LLM CALL   %s/%s — calling %s", ticker, form_type, analysis_type, model_id)
    yield _sse({"type": "progress", "step": step, "status": "processing"})
    t0 = time.monotonic()
    try:
        analysis = await asyncio.to_thread(
            _run_llm_analysis, ticker, analysis_type, raw_data, model_id, api_key,
        )
        await save_filing_analysis(
            ticker, form_type, accession, analysis_type, json.dumps(analysis),
        )
        duration = round(time.monotonic() - t0, 1)
        logger.info("[%s] LLM DONE   %s/%s (%.1fs)", ticker, form_type, analysis_type, duration)
        yield _sse({"type": "progress", "step": step, "status": "done", "duration": duration})
        yield ("result", analysis)
    except Exception as e:
        logger.error("[%s] LLM FAILED %s/%s: %s", ticker, form_type, analysis_type, e)
        yield _sse({"type": "progress", "step": step, "status": "failed"})
        yield ("result", None)


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


def _fetch_filing_data(ticker: str, sec_header: str) -> dict[str, Any]:
    """Synchronous SEC data fetching — runs in a thread pool.

    Returns raw filing data (metadata, section text) without LLM analysis.
    Each filing type fails independently.

    For foreign private issuers (BP, BABA, TSM), falls back to 20-F when
    no 10-K is available. SECDataRetrieval handles 20-F via the same
    get_section() / get_risk_factors_raw() / get_mda_raw() interface.

    6-K is NOT a 10-Q replacement — edgartools' SixK class is a press
    release wrapper without structured section access.
    """
    from agents.sec_workflow.get_SEC_data import SECDataRetrieval

    retriever = SECDataRetrieval(ticker, sec_header)
    result: dict[str, Any] = {"ticker": ticker}

    # --- 10-K (or 20-F for foreign filers) ---
    try:
        retriever.get_tenk_filing()
        meta = retriever._tenk_metadata
        if meta:
            result["tenk"] = {
                "metadata": {
                    **meta.to_dict(),
                    "edgar_url": _build_edgar_url(meta.cik, meta.accession),
                    "form_type": "10-K",
                },
                "risk_raw": retriever.get_risk_factors_raw("10-K"),
                "mda_raw": retriever.get_mda_raw("10-K"),
                "balance_sheet_raw": retriever.extract_balance_sheet_as_str("tenk"),
                "business_raw": retriever.get_business_raw(),
                "cyber_raw": retriever.get_cybersecurity_raw(),
                "legal_raw": retriever.get_legal_proceedings_raw(),
                "market_risk_raw": retriever.get_market_risk_raw("10-K"),
                "income_stmt_raw": retriever.get_income_statement("10-K"),
                "cashflow_raw": retriever.get_cashflow_statement("10-K"),
            }
    except (ValueError, Exception):
        # Foreign filers (BP, BABA, TSM) file 20-F instead of 10-K.
        # SECDataRetrieval.get_section("20-F", item) uses TwentyF's named
        # property accessors (.risk_factors, .management_discussion).
        try:
            retriever.get_twentyf_filing()
            meta = retriever._twentyf_metadata
            if meta:
                result["tenk"] = {
                    "metadata": {
                        **meta.to_dict(),
                        "edgar_url": _build_edgar_url(meta.cik, meta.accession),
                        "form_type": "20-F",
                    },
                    "risk_raw": retriever.get_risk_factors_raw("20-F"),
                    "mda_raw": retriever.get_mda_raw("20-F"),
                    "balance_sheet_raw": {},
                }
        except (ValueError, Exception):
            pass  # No 10-K or 20-F available

    # --- 10-Q ---
    # Foreign filers use 6-K, but edgartools' SixK class only wraps press
    # releases — no structured section access. Quarterly section will be
    # empty for foreign filers until edgartools adds 6-K parsing.
    try:
        retriever.get_tenq_filing()
        meta = retriever._tenq_metadata
        if meta:
            result["tenq"] = {
                "metadata": {
                    **meta.to_dict(),
                    "edgar_url": _build_edgar_url(meta.cik, meta.accession),
                },
                "risk_raw": retriever.get_risk_factors_raw("10-Q"),
                "mda_raw": retriever.get_mda_raw("10-Q"),
                "income_stmt_raw": retriever.get_income_statement("10-Q"),
                "cashflow_raw": retriever.get_cashflow_statement("10-Q"),
            }
    except (ValueError, Exception):
        pass  # No 10-Q available

    # --- 8-K (earnings or material event) ---
    # Always fetch the overview first; branch on `has_earnings` to dispatch to
    # the right analyzer downstream. Earnings (Item 2.02 + parseable EX-99.1)
    # goes to the earnings analyzer; everything else goes to the material-event
    # analyzer (leadership changes, M&A, cyber incidents, Reg FD, etc.).
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
            if overview.get("has_earnings"):
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
    keys: ApiKeys = Depends(get_api_keys),
):
    """Return LLM-analyzed SEC filing summaries for the Filings tab.

    First call for a ticker runs LLM analysis (~5-15s, costs ~$0.005).
    Subsequent calls return instantly from SQLite (keyed by accession
    number — automatic cache invalidation when new filings are published).

    Requires an LLM API key and SEC header (via headers or env vars).
    """
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ticker = ticker.upper()

    # Resolve model + API key
    from agents.model_registry import get_model, get_default_model

    model_id = keys.model_id or get_default_model().id
    model = get_model(model_id) or get_default_model()
    try:
        api_key = keys.require_provider_key(model.provider)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{model.provider.replace('_', ' ').title()} API key required for filing analysis",
        )

    # Resolve SEC header
    sec_header = keys.sec_header
    if not sec_header:
        raise HTTPException(
            status_code=400,
            detail="SEC header required (set X-Sec-Header or SEC_HEADER env var)",
        )

    # Step 1: Fetch raw filing data from EDGAR (no LLM, ~1-3s)
    logger.info("[%s] Fetching raw filings from EDGAR...", ticker)
    t0 = time.monotonic()
    filing_data = await asyncio.to_thread(_fetch_filing_data, ticker, sec_header)
    logger.info("[%s] EDGAR fetch complete (%.1fs)", ticker, time.monotonic() - t0)

    # Step 2: For each filing section, check DB cache then run LLM if needed
    from api.db import get_filing_analysis, save_filing_analysis

    async def _cached_analysis(
        form_type: str, accession: str, analysis_type: str,
        raw_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Check DB cache, run LLM on miss, save result. Returns None on error."""
        cached = await get_filing_analysis(ticker, form_type, accession, analysis_type)
        if cached:
            logger.info("[%s] CACHE HIT  %s/%s (accession: %s)", ticker, form_type, analysis_type, accession)
            return json.loads(cached["analysis_json"])
        logger.info("[%s] LLM CALL   %s/%s (accession: %s) — calling %s", ticker, form_type, analysis_type, accession, model.id)
        t_llm = time.monotonic()
        try:
            analysis = await asyncio.to_thread(
                _run_llm_analysis, ticker, analysis_type,
                raw_data, model.id, api_key,
            )
            await save_filing_analysis(
                ticker, form_type, accession, analysis_type, json.dumps(analysis),
            )
            logger.info("[%s] LLM DONE   %s/%s (%.1fs) — saved to DB", ticker, form_type, analysis_type, time.monotonic() - t_llm)
            return analysis
        except Exception as e:
            logger.error("[%s] LLM FAILED %s/%s: %s", ticker, form_type, analysis_type, e)
            return None

    # Build all analysis coroutines upfront so they can run in parallel.
    # Each entry is (section_key, analysis_key, coroutine).
    analysis_tasks: list[tuple[str, str, Any]] = []

    if "tenk" in filing_data:
        tenk = filing_data["tenk"]
        acc = tenk["metadata"].get("accession", "")
        for atype, raw_key in [("risk_10k", "risk_raw"), ("mda_10k", "mda_raw")]:
            analysis_tasks.append(("tenk", atype, _cached_analysis("10-K", acc, atype, tenk[raw_key])))

        balance_input: dict[str, Any] = {"tenk": tenk.get("balance_sheet_raw", {})}
        if "tenq" in filing_data:
            tenq_bs = filing_data["tenq"].get("balance_sheet_raw")
            if tenq_bs:
                balance_input["tenq"] = tenq_bs
        analysis_tasks.append(("tenk", "balance", _cached_analysis("10-K", acc, "balance", balance_input)))

    if "tenq" in filing_data:
        tenq = filing_data["tenq"]
        acc = tenq["metadata"].get("accession", "")
        for atype, raw_key in [("risk_10q", "risk_raw"), ("mda_10q", "mda_raw")]:
            analysis_tasks.append(("tenq", atype, _cached_analysis("10-Q", acc, atype, tenq[raw_key])))

    eightk = filing_data.get("eightk", {})
    eightk_kind = eightk.get("kind", "none")
    if eightk_kind in ("earnings", "event"):
        acc = eightk.get("metadata", {}).get("accession", "")
        analysis_tasks.append(
            ("eightk", "analysis", _cached_analysis("8-K", acc, eightk_kind, eightk["raw"]))
        )

    # Run all LLM analyses concurrently — on cache hit they return instantly,
    # on cache miss the LLM calls run in parallel threads via asyncio.to_thread.
    results = await asyncio.gather(*(coro for _, _, coro in analysis_tasks))

    # Assemble response from parallel results
    response: dict[str, Any] = {"ticker": ticker}

    if "tenk" in filing_data:
        response["tenk"] = {"metadata": filing_data["tenk"]["metadata"]}
    if "tenq" in filing_data:
        response["tenq"] = {"metadata": filing_data["tenq"]["metadata"]}

    for (section, key, _), result in zip(analysis_tasks, results):
        if result is None:
            continue
        if section in ("tenk", "tenq"):
            response[section][key] = result
        elif section == "eightk":
            response["eightk"] = {
                "kind": eightk_kind,
                "metadata": eightk.get("metadata", {}),
                "analysis": result,
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
    try:
        api_key = keys.require_provider_key(model.provider)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{model.provider.replace('_', ' ').title()} API key required for filing analysis",
        )

    sec_header = keys.sec_header
    if not sec_header:
        raise HTTPException(
            status_code=400,
            detail="SEC header required (set X-Sec-Header or SEC_HEADER env var)",
        )

    # Per-IP concurrency guard — prevents a single client from opening
    # many concurrent streams and exhausting the thread pool or API credits.
    client_ip = request.client.host if request.client else "unknown"
    semaphore = _stream_semaphores[client_ip]
    if semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent filing streams (max {_MAX_CONCURRENT_STREAMS})",
        )

    async def generate():
        # Acquire a concurrency slot for the duration of this stream.
        # Released when the generator finishes (normal exit or client disconnect).
        await semaphore.acquire()
        try:
            # Step 1: Fetch raw filings from EDGAR
            yield _sse({"type": "progress", "step": "edgar_fetch", "status": "fetching"})
            try:
                t0 = time.monotonic()
                filing_data = await asyncio.to_thread(_fetch_filing_data, ticker, sec_header)
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

            # Run all LLM sections concurrently using a shared queue.
            # Each task pushes SSE events (progress + section) to the queue
            # as they happen; the outer generator yields them to the client
            # in real time.  This cuts wall-clock time from the *sum* of all
            # LLM calls (~90s worst case) to the *max* of them (~15s).
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def section_task(form_type: str, accession: str, analysis_type: str, raw_data: dict):
                """Run one analysis and push events to the shared queue."""
                result = None
                async for item in _stream_section(form_type, accession, analysis_type, raw_data, ticker, model.id, api_key):
                    if isinstance(item, tuple):
                        result = item[1]
                    else:
                        await queue.put(item)
                if result is not None:
                    await queue.put(_sse({"type": "section", "form": form_type, "key": analysis_type, "data": result}))

            # Collect all independent section tasks
            tasks: list[Any] = []

            if "tenk" in filing_data:
                tenk = filing_data["tenk"]
                accession = tenk["metadata"].get("accession", "")

                # Existing financial statement sections
                for analysis_type, raw_key in [("risk_10k", "risk_raw"), ("mda_10k", "mda_raw")]:
                    tasks.append(section_task("10-K", accession, analysis_type, tenk[raw_key]))

                balance_input: dict[str, Any] = {"tenk": tenk.get("balance_sheet_raw", {})}
                if "tenq" in filing_data:
                    tenq_bs = filing_data["tenq"].get("balance_sheet_raw")
                    if tenq_bs:
                        balance_input["tenq"] = tenq_bs
                tasks.append(section_task("10-K", accession, "balance", balance_input))

                # Newly surfaced 10-K text sections (Item 1, 1C, 3, 7A)
                for analysis_type, raw_key in [
                    ("business", "business_raw"),
                    ("cybersecurity", "cyber_raw"),
                    ("legal", "legal_raw"),
                    ("market_risk", "market_risk_raw"),
                ]:
                    raw = tenk.get(raw_key)
                    if raw and raw.get("found"):
                        tasks.append(section_task("10-K", accession, analysis_type, raw))

                # Income statement — combines 10-K + 10-Q XBRL data
                if tenk.get("income_stmt_raw"):
                    income_input: dict[str, Any] = {
                        "tenk": tenk["income_stmt_raw"],
                        "tenk_metadata": tenk["metadata"],
                    }
                    if "tenq" in filing_data and filing_data["tenq"].get("income_stmt_raw"):
                        income_input["tenq"] = filing_data["tenq"]["income_stmt_raw"]
                        income_input["tenq_metadata"] = filing_data["tenq"]["metadata"]
                    tasks.append(section_task("10-K", accession, "income_stmt", income_input))

                # Cash flow — combines 10-K + 10-Q XBRL data
                if tenk.get("cashflow_raw"):
                    cashflow_input: dict[str, Any] = {
                        "tenk": tenk["cashflow_raw"],
                        "tenk_metadata": tenk["metadata"],
                    }
                    if "tenq" in filing_data and filing_data["tenq"].get("cashflow_raw"):
                        cashflow_input["tenq"] = filing_data["tenq"]["cashflow_raw"]
                        cashflow_input["tenq_metadata"] = filing_data["tenq"]["metadata"]
                    tasks.append(section_task("10-K", accession, "cashflow", cashflow_input))

            if "tenq" in filing_data:
                tenq = filing_data["tenq"]
                accession = tenq["metadata"].get("accession", "")
                for analysis_type, raw_key in [("risk_10q", "risk_raw"), ("mda_10q", "mda_raw")]:
                    tasks.append(section_task("10-Q", accession, analysis_type, tenq[raw_key]))

            if eightk_kind in ("earnings", "event"):
                accession = eightk.get("metadata", {}).get("accession", "")
                tasks.append(section_task("8-K", accession, eightk_kind, eightk["raw"]))

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
