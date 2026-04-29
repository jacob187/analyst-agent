import json
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any, Optional

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from agents.sec_workflow.sec_llm_models import SECDocumentProcessor


def _dump_analysis_json(result: Any) -> str:
    """Serialize a cached analysis dict (or error) to JSON for the synthesizer.

    Each LLM-analysis tool caches ``analysis.model_dump()`` as a dict. Returning
    the full dict as JSON preserves every Pydantic field (financial_highlights,
    liquidity_analysis, sentiment_analysis, filing_metadata, etc.) instead of
    cherry-picking 2-5 fields into a hand-formatted f-string. The synthesizer
    can then reconcile across tools by inspecting structured fields like
    ``filing_metadata`` rather than parsing prose.

    ``default=str`` handles non-serializable types (numpy scalars, datetimes)
    that may slip through ``model_dump()``.
    """
    if isinstance(result, dict):
        return json.dumps(result, indent=2, default=str)
    return json.dumps({"value": str(result)}, indent=2)


# Global cache for shared SEC data retrievers, processors, and processed outputs
_shared_retrievers: Dict[str, SECDataRetrieval] = {}
_shared_processors: Dict[str, SECDocumentProcessor] = {}
_processed_cache: Dict[str, Dict[str, Any]] = {}


def _get_shared_retriever(ticker: str, sec_header: str) -> SECDataRetrieval:
    """Get or create a shared SEC data retriever for the ticker.

    The retriever is created once per ticker and cached for the process
    lifetime.  Filing availability is NOT checked here -- tools that need
    a 10-K should call ``_require_10k()`` before accessing 10-K data.
    """
    if ticker not in _shared_retrievers:
        print(f"Making single SEC API call for {ticker}...")
        retriever = SECDataRetrieval(ticker, sec_header)
        _shared_retrievers[ticker] = retriever
        _processed_cache[ticker] = {}
    return _shared_retrievers[ticker]


def _require_10k(retriever: SECDataRetrieval) -> None:
    """Raise ValueError if the company has no 10-K filing.

    Called lazily by tools that need 10-K data, rather than at retriever
    creation time.  This allows 8-K-only queries to work for companies
    that don't file 10-Ks (SPACs, foreign filers, etc.).
    """
    availability = retriever.check_filing_availability()
    if not availability["has_10k"]:
        raise ValueError(
            f"No 10-K filing found for {availability['company_name']} ({retriever.ticker}). "
            f"This company may file different forms (e.g., N-CSR for investment trusts)."
        )


def _get_shared_processor(ticker: str, llm: BaseChatModel) -> SECDocumentProcessor:
    """Get or create a shared SEC document processor for the ticker."""
    processor_key = f"{ticker}_{id(llm)}"
    if processor_key not in _shared_processors:
        _shared_processors[processor_key] = SECDocumentProcessor(llm)
    return _shared_processors[processor_key]


def _get_cache_key(ticker: str, analysis_type: str) -> str:
    """Generate cache key for processed analysis."""
    return f"{ticker}_{analysis_type}"


def _is_substantive(text: str) -> bool:
    """Return True if text has real content rather than 10-Q boilerplate."""
    boilerplate_phrases = [
        "no material changes",
        "not materially changed",
        "no significant changes",
        "previously disclosed",
        "incorporated by reference",
    ]
    lower = text.lower().strip()
    return bool(lower) and (len(lower) >= 500 or not any(phrase in lower for phrase in boilerplate_phrases))


def _fetch_best_section(fetch_fn) -> Dict[str, Any]:
    """Try 10-Q first; fall back to 10-K when 10-Q is boilerplate or unavailable."""
    result = fetch_fn("10-Q")
    if result.get("found") and _is_substantive(result.get("text", "")):
        return result
    return fetch_fn("10-K")


def _tool_raw_risk_factors(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return raw Risk Factors text, preferring 10-Q (most recent) with 10-K fallback."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        result = _fetch_best_section(retriever.get_risk_factors_raw)
        if not result.get("found"):
            return f"Risk Factors section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve risk factors for {ticker}: {e}"


def _tool_risk_factors_summary(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return summarized Risk Factors, preferring 10-Q with 10-K fallback (cached)."""
    cache_key = _get_cache_key(ticker, "risk_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker, sec_header)
            _require_10k(retriever)
            processor = _get_shared_processor(ticker, llm)
            risk_data = _fetch_best_section(retriever.get_risk_factors_raw)
            if not risk_data.get("found", False):
                return f"Risk Factors section not found for {ticker}."
            analysis = processor.analyze_risk_factors(ticker, risk_data)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze risk factors for {ticker}: {e}"
    return _dump_analysis_json(_processed_cache[ticker][cache_key])


def _tool_raw_mda(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return raw MD&A text, preferring 10-Q (most recent) with 10-K fallback."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        result = _fetch_best_section(retriever.get_mda_raw)
        if not result.get("found"):
            return f"Management Discussion section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve MD&A for {ticker}: {e}"


def _tool_mda_summary(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return summarized MD&A, preferring 10-Q with 10-K fallback (cached)."""
    cache_key = _get_cache_key(ticker, "mda_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker, sec_header)
            _require_10k(retriever)
            processor = _get_shared_processor(ticker, llm)
            mda_data = _fetch_best_section(retriever.get_mda_raw)
            if not mda_data.get("found", False):
                return f"Management Discussion section not found for {ticker}."
            analysis = processor.analyze_mda(ticker, mda_data)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze MD&A for {ticker}: {e}"
    return _dump_analysis_json(_processed_cache[ticker][cache_key])


def _tool_raw_balance_sheets(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return availability list of balance sheet JSON sections for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        result = retriever.extract_balance_sheet_as_json()
        if isinstance(result, dict) and "error" not in result:
            return f"Balance sheet data available for: {list(result.keys())}"
        return str(result)
    except Exception as e:
        return f"Failed to retrieve balance sheets for {ticker}: {e}"


def _tool_balance_sheet_summary(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return summarized balance sheet analysis for the given ticker (cached)."""
    cache_key = _get_cache_key(ticker, "balance_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker, sec_header)
            _require_10k(retriever)
            processor = _get_shared_processor(ticker, llm)
            balance_data = retriever.extract_balance_sheet_as_json()
            analysis = processor.analyze_balance_sheet(
                ticker,
                balance_data.get("tenk", {}),
                balance_data.get("tenq", {}),
            )
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze balance sheet for {ticker}: {e}"
    return _dump_analysis_json(_processed_cache[ticker][cache_key])


def _tool_business_overview(ticker: str, sec_header: str = "") -> str:
    """Return raw Business Overview text (10-K Item 1)."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        result = retriever.get_business_raw()
        if not result.get("found"):
            return f"Business overview not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve business overview for {ticker}: {e}"


def _tool_cybersecurity_disclosure(ticker: str, sec_header: str = "") -> str:
    """Return Cybersecurity risk management disclosure (10-K Item 1C)."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        result = retriever.get_cybersecurity_raw()
        if not result.get("found"):
            return f"Cybersecurity disclosure not found for {ticker} (may not be present in older filings)."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve cybersecurity disclosure for {ticker}: {e}"


def _tool_legal_proceedings(ticker: str, sec_header: str = "") -> str:
    """Return Legal Proceedings text (10-K Item 3)."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        result = retriever.get_legal_proceedings_raw()
        if not result.get("found"):
            return f"Legal proceedings section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve legal proceedings for {ticker}: {e}"


def _tool_complete_10k_text(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return list of available major 10-K sections for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        _require_10k(retriever)
        mda_data = retriever.get_mda_raw("10-K")
        risk_data = retriever.get_risk_factors_raw("10-K")
        
        sections_available = []
        if mda_data.get("found", False):
            sections_available.append("Management Discussion & Analysis")
        if risk_data.get("found", False):
            sections_available.append("Risk Factors")
        
        if not sections_available:
            return f"No 10-K sections found for {ticker}"
        
        return f"Complete 10-K sections available for {ticker}: {', '.join(sections_available)}"
    except Exception as e:
        return f"Failed to retrieve complete 10-K for {ticker}: {e}"


def _tool_all_summaries(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return a comprehensive overview across risks, MD&A, and financials.

    The three analyses are independent (different filing sections, separate LLM calls),
    so we run them concurrently. This cuts wall-clock time from ~12s (3 sequential
    LLM calls) to ~5s (limited by the slowest single call).
    """
    with ThreadPoolExecutor(max_workers=3) as pool:
        risk_future = pool.submit(_tool_risk_factors_summary, ticker, llm, sec_header)
        mda_future = pool.submit(_tool_mda_summary, ticker, llm, sec_header)
        balance_future = pool.submit(_tool_balance_sheet_summary, ticker, llm, sec_header)

        risk_summary = risk_future.result()
        mda_summary = mda_future.result()
        balance_summary = balance_future.result()

    summary = f"Comprehensive Analysis for {ticker}:\n\n"
    summary += f"=== RISK ANALYSIS ===\n{risk_summary}\n\n"
    summary += f"=== MANAGEMENT OUTLOOK ===\n{mda_summary}\n\n"
    summary += f"=== FINANCIAL HEALTH ===\n{balance_summary}\n"
    return summary


# ── 8-K tool functions ────────────────────────────────────────────────────────


def _tool_8k_overview(ticker: str, sec_header: str = "") -> str:
    """Return overview of the latest 8-K filing: items, event type, dates."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        result = retriever.get_8k_overview()
        if not result.get("found"):
            return f"No 8-K filing found for {ticker}."
        items = result.get("items", [])
        return (
            f"Latest 8-K for {ticker}:\n"
            f"Event Type: {result.get('content_type', 'unknown')}\n"
            f"Items: {', '.join(items)}\n"
            f"Date of Report: {result.get('date_of_report', 'Unknown')}\n"
            f"Has Earnings: {result.get('has_earnings', False)}\n\n"
            f"{result.get('context', '')}"
        )
    except Exception as e:
        return f"Failed to retrieve 8-K overview for {ticker}: {e}"


def _tool_8k_item(ticker: str, item: str, sec_header: str = "") -> str:
    """Return raw text of a specific 8-K item (e.g. '2.02', '1.01')."""
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        result = retriever.get_8k_item(item)
        if not result.get("found"):
            return f"8-K Item {item} not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve 8-K item {item} for {ticker}: {e}"


def _tool_earnings_summary(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Return structured earnings analysis from 8-K Item 2.02 (cached)."""
    cache_key = _get_cache_key(ticker, "earnings_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker, sec_header)
            processor = _get_shared_processor(ticker, llm)
            earnings_data = retriever.get_earnings_data()
            if not earnings_data.get("has_earnings"):
                return (
                    f"No earnings data in latest 8-K for {ticker}. "
                    f"Reason: {earnings_data.get('reason', 'Unknown')}"
                )
            analysis = processor.analyze_earnings(ticker, earnings_data)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze earnings for {ticker}: {e}"
    return _dump_analysis_json(_processed_cache[ticker][cache_key])


def _tool_material_event_summary(
    ticker: str,
    llm: BaseChatModel,
    sec_header: str = "",
    overview: Optional[Dict[str, Any]] = None,
) -> str:
    """Return structured analysis of non-earnings 8-K material event (cached).

    If ``overview`` is provided (the dispatcher does this), reuse it instead
    of re-fetching from the retriever — saves one round-trip + dict construction
    per material-event 8-K query.
    """
    cache_key = _get_cache_key(ticker, "material_event_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker, sec_header)
            processor = _get_shared_processor(ticker, llm)
            if overview is None:
                overview = retriever.get_8k_overview()
            if not overview.get("found"):
                return f"No 8-K filing found for {ticker}."
            # Get the text of the primary (first) item for analysis
            items = overview.get("items", [])
            primary_item = items[0] if items else None
            item_text = ""
            if primary_item:
                item_result = retriever.get_8k_item(primary_item)
                item_text = item_result.get("text", "") if item_result.get("found") else ""
            event_data = {
                "content_type": overview.get("content_type", "other"),
                "items": items,
                "context": overview.get("context", ""),
                "text": item_text,
                "metadata": overview.get("metadata", {}),
            }
            analysis = processor.analyze_material_event(ticker, event_data)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze material event for {ticker}: {e}"
    return _dump_analysis_json(_processed_cache[ticker][cache_key])


def _tool_analyze_latest_8k(ticker: str, llm: BaseChatModel, sec_header: str = "") -> str:
    """Analyze the latest 8-K, dispatching to the right analyzer by item code.

    Earnings releases (Item 2.02 with a parseable EX-99.1 press release) go to
    the earnings analyzer; everything else (leadership changes, agreements,
    cybersecurity incidents, Reg FD disclosures) goes to the material event
    analyzer. Picking exactly one of the two analyzers prevents the planner
    from double-fetching overlapping tools and emitting contradictory event
    classifications for the same filing.
    """
    try:
        retriever = _get_shared_retriever(ticker, sec_header)
        overview = retriever.get_8k_overview()
        if not overview.get("found"):
            return json.dumps(
                {"found": False, "reason": f"No 8-K filing found for {ticker}"},
                indent=2,
            )
    except Exception as e:
        return json.dumps(
            {"error": str(e), "tool": "analyze_latest_8k", "ticker": ticker},
            indent=2,
        )

    if overview.get("has_earnings"):
        # The earnings analyzer fetches its data via ``get_earnings_data()``,
        # not from the overview, so there's nothing to forward here.
        return _tool_earnings_summary(ticker, llm, sec_header)
    # Forward the overview so the material-event analyzer doesn't re-fetch it.
    return _tool_material_event_summary(ticker, llm, sec_header, overview=overview)


def create_sec_tools(ticker: str, llm: BaseChatModel, sec_header: str = "") -> tuple[list, str]:
    """Return the list of SEC tools bound to a specific ticker and LLM.

    Each tool is a plain function (no bound instance methods) wrapped as a Tool.

    Args:
        ticker: Company ticker symbol
        llm: LLM for document analysis
        sec_header: SEC EDGAR identity header (e.g. "Name email@example.com")
    """
    llm_id = f"{ticker}_{id(llm)}"

    tools: list[Tool] = [
        Tool.from_function(
            name="get_raw_risk_factors",
            description="Get the complete raw text of Risk Factors section from 10-K filing.",
            func=lambda query="": _tool_raw_risk_factors(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_risk_factors_summary",
            description="Get structured analysis of Risk Factors with sentiment and key risks.",
            func=lambda query="": _tool_risk_factors_summary(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_raw_management_discussion",
            description="Get raw text of Management Discussion & Analysis (MD&A).",
            func=lambda query="": _tool_raw_mda(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_mda_summary",
            description="Get structured MD&A summary with sentiment, outlook, and key points.",
            func=lambda query="": _tool_mda_summary(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_raw_balance_sheets",
            description="Get availability of raw balance sheet JSON data (10-K and 10-Q).",
            func=lambda query="": _tool_raw_balance_sheets(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_balance_sheet_summary",
            description="Get balance sheet summary with key metrics and red flags.",
            func=lambda query="": _tool_balance_sheet_summary(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_complete_10k_text",
            description="Get list of available 10-K sections retrieved as raw text.",
            func=lambda query="": _tool_complete_10k_text(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_business_overview",
            description="Get the Company Business section (10-K Item 1): products, services, segments, and market overview.",
            func=lambda query="": _tool_business_overview(ticker, sec_header),
        ),
        Tool.from_function(
            name="get_cybersecurity_disclosure",
            description="Get the Cybersecurity risk management and governance disclosure (10-K Item 1C, SEC-mandated since 2023).",
            func=lambda query="": _tool_cybersecurity_disclosure(ticker, sec_header),
        ),
        Tool.from_function(
            name="get_legal_proceedings",
            description="Get Legal Proceedings section (10-K Item 3): significant pending litigation, regulatory actions, and legal risks.",
            func=lambda query="": _tool_legal_proceedings(ticker, sec_header),
        ),
        Tool.from_function(
            name="get_all_summaries",
            description="Get a comprehensive overview across risks, MD&A, and financials.",
            func=lambda query="": _tool_all_summaries(ticker, llm, sec_header),
        ),
        # ── 8-K tools ────────────────────────────────────────────────────
        # Dispatcher pattern: a single planner-visible tool routes internally
        # to either the earnings analyzer or the material event analyzer.
        # Replaces the older overview / earnings_summary / material_event_summary
        # trio, which let the planner pick overlapping tools that classified
        # the same filing differently.
        Tool.from_function(
            name="analyze_latest_8k",
            description=(
                "Get structured analysis of the latest 8-K filing. Automatically "
                "routes to earnings analysis (Item 2.02 with a press release) or "
                "material event analysis (leadership changes, agreements, "
                "cybersecurity incidents, Reg FD) based on the filing's contents. "
                "Returns sentiment, key points, impact assessment, and filing "
                "metadata as JSON."
            ),
            func=lambda query="": _tool_analyze_latest_8k(ticker, llm, sec_header),
        ),
        Tool.from_function(
            name="get_8k_item",
            description=(
                "Get raw text of a specific 8-K item by number (e.g. '2.02' for earnings, "
                "'1.01' for material agreements, '5.02' for leadership changes). "
                "Pass the item number as the query."
            ),
            func=lambda item="2.02": _tool_8k_item(ticker, item, sec_header),
        ),
    ]

    return tools, llm_id
