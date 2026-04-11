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
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import ApiKeys, get_api_keys

logger = logging.getLogger("analyst.filings")

router = APIRouter(prefix="/api/company")

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")

# Server-side cache: {ticker: (monotonic_timestamp, payload)}
# Values are JSON-round-tripped dicts to guarantee serializability.
_profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_profile_locks: dict[str, asyncio.Lock] = {}
_PROFILE_CACHE_TTL = 300  # 5 minutes


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

    # Split company dict into "company" (metadata) and "metrics" (financials)
    company_metadata = {
        "name": company.get("shortName"),
        "sector": company.get("sector"),
        "industry": company.get("industry"),
        "country": company.get("country"),
        "website": company.get("website"),
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

    return {
        "ticker": ticker,
        "company": company_metadata,
        "metrics": metrics,
        "quote": quote,
        "earnings": earnings,
        "technicals": technicals,
        "patterns": patterns,
        "regime": regime,
    }


@router.get("/{ticker}/profile")
async def get_company_profile(ticker: str):
    """Return aggregated company profile for the dashboard Overview tab.

    Combines yfinance metadata, live quote, technical indicators, chart
    patterns, and market regime into a single response. No LLM calls —
    pure data aggregation, typically responds in < 1 second.
    """
    if not _TICKER_RE.match(ticker):
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

        # Run synchronous data fetching in a thread to avoid blocking the event loop
        raw_payload = await asyncio.to_thread(_build_profile, ticker)

        # Round-trip through JSON to guarantee serializability (numpy floats, etc.)
        payload = json.loads(json.dumps(raw_payload, default=str))
        _profile_cache[ticker] = (time.monotonic(), payload)

    return JSONResponse(
        content=payload,
        headers={"Cache-Control": f"public, max-age={_PROFILE_CACHE_TTL}"},
    )


# =============================================================================
# Filings endpoint — LLM-powered SEC analysis with DB-first caching
# =============================================================================

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
                },
                "risk_raw": retriever.get_risk_factors_raw("10-K"),
                "mda_raw": retriever.get_mda_raw("10-K"),
                "balance_sheet_raw": retriever.extract_balance_sheet_as_str("tenk"),
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
            }
    except (ValueError, Exception):
        pass  # No 10-Q available

    # --- 8-K earnings ---
    try:
        earnings = retriever.get_earnings_data()
        if earnings.get("has_earnings"):
            eightk_meta = retriever._eightk_metadata
            result["earnings_raw"] = {
                **earnings,
                "metadata": {
                    **(eightk_meta.to_dict() if eightk_meta else {}),
                    "edgar_url": _build_edgar_url(
                        eightk_meta.cik, eightk_meta.accession
                    ) if eightk_meta else "",
                },
            }
        else:
            result["earnings_raw"] = earnings
    except (ValueError, Exception):
        result["earnings_raw"] = {"has_earnings": False, "reason": "No 8-K available"}

    return result


def _run_llm_analysis(
    ticker: str, analysis_type: str, raw_data: dict[str, Any],
    model_id: str, api_key: str,
) -> dict[str, Any]:
    """Run a single LLM analysis and return the Pydantic model as a dict.

    analysis_type is one of: 'risk_10k', 'mda_10k', 'balance', 'risk_10q',
    'mda_10q', 'earnings'.
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
    if not _TICKER_RE.match(ticker):
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

    response: dict[str, Any] = {"ticker": ticker}

    # --- 10-K analyses ---
    if "tenk" in filing_data:
        tenk = filing_data["tenk"]
        accession = tenk["metadata"].get("accession", "")
        tenk_result: dict[str, Any] = {"metadata": tenk["metadata"]}

        for analysis_type, raw_key in [
            ("risk_10k", "risk_raw"),
            ("mda_10k", "mda_raw"),
        ]:
            result = await _cached_analysis("10-K", accession, analysis_type, tenk[raw_key])
            if result is not None:
                tenk_result[analysis_type] = result

        # Balance sheet (uses both 10-K and 10-Q data)
        balance_input: dict[str, Any] = {"tenk": tenk.get("balance_sheet_raw", {})}
        if "tenq" in filing_data:
            tenq_bs = filing_data["tenq"].get("balance_sheet_raw")
            if tenq_bs:
                balance_input["tenq"] = tenq_bs
        result = await _cached_analysis("10-K", accession, "balance", balance_input)
        if result is not None:
            tenk_result["balance"] = result

        response["tenk"] = tenk_result

    # --- 10-Q analyses ---
    if "tenq" in filing_data:
        tenq = filing_data["tenq"]
        accession = tenq["metadata"].get("accession", "")
        tenq_result: dict[str, Any] = {"metadata": tenq["metadata"]}

        for analysis_type, raw_key in [
            ("risk_10q", "risk_raw"),
            ("mda_10q", "mda_raw"),
        ]:
            result = await _cached_analysis("10-Q", accession, analysis_type, tenq[raw_key])
            if result is not None:
                tenq_result[analysis_type] = result

        response["tenq"] = tenq_result

    # --- 8-K earnings ---
    earnings_raw = filing_data.get("earnings_raw", {})
    if earnings_raw.get("has_earnings"):
        accession = earnings_raw.get("metadata", {}).get("accession", "")
        earnings_analysis = await _cached_analysis("8-K", accession, "earnings", earnings_raw)
        response["earnings"] = {
            "has_earnings": True,
            "metadata": earnings_raw.get("metadata", {}),
            "analysis": earnings_analysis,
        }
    else:
        response["earnings"] = {
            "has_earnings": False,
            "reason": earnings_raw.get("reason", "No earnings data available"),
        }

    return JSONResponse(content=response)
