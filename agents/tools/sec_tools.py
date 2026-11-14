import asyncio
import json
from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Callable, Dict, Any

from agents.sec_workflow.analysis_cache import (
    FilingRef,
    eightk_input,
    get_or_analyze,
    get_retriever,
    latest_filing,
    run_analysis,
)
from agents.sec_workflow.get_SEC_data import SECDataRetrieval


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


def _best_form(fetch_fn, try_10q: bool = True) -> tuple[str, Dict[str, Any]]:
    """``(form, section)``: the 10-Q section when substantive, else the 10-K's."""
    if try_10q:
        result = fetch_fn("10-Q")
        if result.get("found") and _is_substantive(result.get("text", "")):
            return "10-Q", result
    return "10-K", fetch_fn("10-K")


def _fetch_best_section(fetch_fn) -> Dict[str, Any]:
    """Try 10-Q first; fall back to 10-K when 10-Q is boilerplate or unavailable."""
    return _best_form(fetch_fn)[1]


class _SectionUnavailable(Exception):
    """The filing section's input could not be extracted, so there is
    nothing to analyse (a cached analysis is still served)."""


async def _latest_10k(ticker: str) -> FilingRef:
    """The latest 10-K ref, or ValueError for companies that file none
    (SPACs, investment trusts, foreign filers)."""
    tenk = await latest_filing(ticker, "10-K")
    if tenk is None:
        raise ValueError(
            f"No 10-K filing found for {ticker}. "
            f"This company may file different forms (e.g., N-CSR for investment trusts)."
        )
    return tenk


async def _serve_analysis(
    ticker: str,
    llm: BaseChatModel,
    filing: FilingRef,
    analysis_type: str,
    load_input: Callable[[], Dict[str, Any]],
) -> str:
    """The analysis of ``filing`` as JSON with its ``filing_reference`` — from
    ``filing_analyses`` (the company page's cache) when present, else
    generated from ``load_input()`` (blocking edgartools, run in a thread)
    and saved there. The LLM runs via ``ainvoke`` in this task's context, so
    the chat run's callbacks (tracing, LLMCallCharger) see the call."""

    async def generate() -> Dict[str, Any]:
        raw = await asyncio.to_thread(load_input)
        return await run_analysis(llm, ticker, analysis_type, raw)

    result = await get_or_analyze(ticker, filing, analysis_type, generate)
    return _dump_analysis_json(result.served())


async def _best_section_summary(
    ticker: str, llm: BaseChatModel, getter: str, stem: str,
) -> str | None:
    """Analysis of the 10-Q section when substantive, else the 10-K's — keyed
    like the company page (``{stem}_10q`` / ``{stem}_10k``). None when the
    section is missing from both."""
    tenk, tenq = await asyncio.gather(_latest_10k(ticker), latest_filing(ticker, "10-Q"))
    retriever = await asyncio.to_thread(get_retriever, ticker)
    form, section = await asyncio.to_thread(
        _best_form, getattr(retriever, getter), tenq is not None,
    )
    if not section.get("found"):
        return None
    if form == "10-Q":
        return await _serve_analysis(ticker, llm, tenq, f"{stem}_10q", lambda: section)
    return await _serve_analysis(ticker, llm, tenk, f"{stem}_10k", lambda: section)


def _tool_raw_risk_factors(ticker: str, llm: BaseChatModel) -> str:
    """Return raw Risk Factors text, preferring 10-Q (most recent) with 10-K fallback."""
    try:
        retriever = get_retriever(ticker)
        _require_10k(retriever)
        result = _fetch_best_section(retriever.get_risk_factors_raw)
        if not result.get("found"):
            return f"Risk Factors section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve risk factors for {ticker}: {e}"


async def _tool_risk_factors_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized Risk Factors, preferring 10-Q with 10-K fallback (cached)."""
    try:
        summary = await _best_section_summary(ticker, llm, "get_risk_factors_raw", "risk")
    except Exception as e:
        return f"Failed to analyze risk factors for {ticker}: {e}"
    return summary or f"Risk Factors section not found for {ticker}."


def _tool_raw_mda(ticker: str, llm: BaseChatModel) -> str:
    """Return raw MD&A text, preferring 10-Q (most recent) with 10-K fallback."""
    try:
        retriever = get_retriever(ticker)
        _require_10k(retriever)
        result = _fetch_best_section(retriever.get_mda_raw)
        if not result.get("found"):
            return f"Management Discussion section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve MD&A for {ticker}: {e}"


async def _tool_mda_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized MD&A, preferring 10-Q with 10-K fallback (cached)."""
    try:
        summary = await _best_section_summary(ticker, llm, "get_mda_raw", "mda")
    except Exception as e:
        return f"Failed to analyze MD&A for {ticker}: {e}"
    return summary or f"Management Discussion section not found for {ticker}."


def _tool_raw_balance_sheets(ticker: str, llm: BaseChatModel) -> str:
    """Return availability list of balance sheet JSON sections for the given ticker."""
    try:
        retriever = get_retriever(ticker)
        _require_10k(retriever)
        result = retriever.extract_balance_sheet_as_json()
        if isinstance(result, dict) and "error" not in result:
            return f"Balance sheet data available for: {list(result.keys())}"
        return str(result)
    except Exception as e:
        return f"Failed to retrieve balance sheets for {ticker}: {e}"


async def _tool_balance_sheet_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized balance sheet analysis for the given ticker (cached)."""

    def load_input() -> Dict[str, Any]:
        # The company page's "balance" input: the latest 10-K's balance sheet,
        # keyed by that 10-K's accession.
        return {"tenk": get_retriever(ticker).extract_balance_sheet_as_str("tenk")}

    try:
        tenk = await _latest_10k(ticker)
        return await _serve_analysis(ticker, llm, tenk, "balance", load_input)
    except Exception as e:
        return f"Failed to analyze balance sheet for {ticker}: {e}"


def _tool_business_overview(ticker: str) -> str:
    """Return raw Business Overview text (10-K Item 1)."""
    try:
        retriever = get_retriever(ticker)
        _require_10k(retriever)
        result = retriever.get_business_raw()
        if not result.get("found"):
            return f"Business overview not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve business overview for {ticker}: {e}"


def _tool_cybersecurity_disclosure(ticker: str) -> str:
    """Return Cybersecurity risk management disclosure (10-K Item 1C)."""
    try:
        retriever = get_retriever(ticker)
        _require_10k(retriever)
        result = retriever.get_cybersecurity_raw()
        if not result.get("found"):
            return f"Cybersecurity disclosure not found for {ticker} (may not be present in older filings)."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve cybersecurity disclosure for {ticker}: {e}"


def _tool_legal_proceedings(ticker: str) -> str:
    """Return Legal Proceedings text (10-K Item 3)."""
    try:
        retriever = get_retriever(ticker)
        _require_10k(retriever)
        result = retriever.get_legal_proceedings_raw()
        if not result.get("found"):
            return f"Legal proceedings section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve legal proceedings for {ticker}: {e}"


def _tool_complete_10k_text(ticker: str, llm: BaseChatModel) -> str:
    """Return list of available major 10-K sections for the given ticker."""
    try:
        retriever = get_retriever(ticker)
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


async def _tool_all_summaries(ticker: str, llm: BaseChatModel) -> str:
    """Return a comprehensive overview across risks, MD&A, and financials.

    The three analyses are independent (different filing sections, separate LLM calls),
    so we run them concurrently. This cuts wall-clock time from ~12s (3 sequential
    LLM calls) to ~5s (limited by the slowest single call). ``asyncio.gather``
    keeps the run's callbacks on each call, so all three are charged and traced.
    """
    risk_summary, mda_summary, balance_summary = await asyncio.gather(
        _tool_risk_factors_summary(ticker, llm),
        _tool_mda_summary(ticker, llm),
        _tool_balance_sheet_summary(ticker, llm),
    )

    summary = f"Comprehensive Analysis for {ticker}:\n\n"
    summary += f"=== RISK ANALYSIS ===\n{risk_summary}\n\n"
    summary += f"=== MANAGEMENT OUTLOOK ===\n{mda_summary}\n\n"
    summary += f"=== FINANCIAL HEALTH ===\n{balance_summary}\n"
    return summary


# ── 8-K tool functions ────────────────────────────────────────────────────────


def _tool_8k_overview(ticker: str) -> str:
    """Return overview of the latest 8-K filing: items, event type, dates."""
    try:
        retriever = get_retriever(ticker)
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


def _tool_8k_item(ticker: str, item: str) -> str:
    """Return raw text of a specific 8-K item (e.g. '2.02', '1.01')."""
    try:
        retriever = get_retriever(ticker)
        result = retriever.get_8k_item(item)
        if not result.get("found"):
            return f"8-K Item {item} not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve 8-K item {item} for {ticker}: {e}"


async def _tool_analyze_latest_8k(ticker: str, llm: BaseChatModel) -> str:
    """Analyze the latest 8-K, dispatching to the right analyzer by item code.

    Earnings releases (Item 2.02 with a parseable EX-99.1 press release) go to
    the earnings analyzer; everything else (leadership changes, agreements,
    cybersecurity incidents, Reg FD disclosures) goes to the material event
    analyzer. Picking exactly one of the two analyzers prevents the planner
    from double-fetching overlapping tools and emitting contradictory event
    classifications for the same filing.

    Shares the company page's cache (analysis types ``earnings`` / ``event``)
    and its rule that an 8-K whose input failed to extract is never analysed.
    """
    try:
        filing = await latest_filing(ticker, "8-K")
        overview: Dict[str, Any] = {"found": False}
        if filing is not None:
            retriever = await asyncio.to_thread(get_retriever, ticker)
            overview = await asyncio.to_thread(retriever.get_8k_overview)
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

    kind = "earnings" if overview.get("has_earnings") else "event"

    def load_input() -> Dict[str, Any]:
        _kind, raw = eightk_input(retriever, overview, overview.get("metadata", {}))
        if kind == "earnings" and raw.get("has_earnings") is False:
            raise _SectionUnavailable(
                f"No earnings data in latest 8-K for {ticker}. "
                f"Reason: {raw.get('reason', 'Unknown')}"
            )
        if kind == "event" and not raw.get("text"):
            raise _SectionUnavailable(
                f"The latest 8-K's item text for {ticker} could not be extracted."
            )
        return raw

    try:
        return await _serve_analysis(ticker, llm, filing, kind, load_input)
    except _SectionUnavailable as e:
        return json.dumps({"found": False, "reason": str(e)}, indent=2)
    except Exception as e:
        return json.dumps(
            {"error": str(e), "tool": "analyze_latest_8k", "ticker": ticker},
            indent=2,
        )


def create_sec_tools(ticker: str, llm: BaseChatModel) -> tuple[list, str]:
    """Return the list of SEC tools bound to a specific ticker and LLM.

    Each tool is a plain function (no bound instance methods) wrapped as a Tool.
    Tools that run an LLM analysis (or touch the DB) are async-only
    (``coroutine=``) — invoke them with ``ainvoke``; raw-text tools are sync.

    Args:
        ticker: Company ticker symbol
        llm: LLM for document analysis
    """
    llm_id = f"{ticker}_{id(llm)}"

    tools: list[Tool] = [
        Tool.from_function(
            name="get_raw_risk_factors",
            description="Get the complete raw text of Risk Factors section from 10-K filing.",
            func=lambda query="": _tool_raw_risk_factors(ticker, llm),
        ),
        Tool.from_function(
            name="get_risk_factors_summary",
            description="Get structured analysis of Risk Factors with sentiment and key risks.",
            func=None,
            coroutine=lambda query="": _tool_risk_factors_summary(ticker, llm),
        ),
        Tool.from_function(
            name="get_raw_management_discussion",
            description="Get raw text of Management Discussion & Analysis (MD&A).",
            func=lambda query="": _tool_raw_mda(ticker, llm),
        ),
        Tool.from_function(
            name="get_mda_summary",
            description="Get structured MD&A summary with sentiment, outlook, and key points.",
            func=None,
            coroutine=lambda query="": _tool_mda_summary(ticker, llm),
        ),
        Tool.from_function(
            name="get_raw_balance_sheets",
            description="Get availability of raw balance sheet JSON data (10-K and 10-Q).",
            func=lambda query="": _tool_raw_balance_sheets(ticker, llm),
        ),
        Tool.from_function(
            name="get_balance_sheet_summary",
            description="Get balance sheet summary with key metrics and red flags.",
            func=None,
            coroutine=lambda query="": _tool_balance_sheet_summary(ticker, llm),
        ),
        Tool.from_function(
            name="get_complete_10k_text",
            description="Get list of available 10-K sections retrieved as raw text.",
            func=lambda query="": _tool_complete_10k_text(ticker, llm),
        ),
        Tool.from_function(
            name="get_business_overview",
            description="Get the Company Business section (10-K Item 1): products, services, segments, and market overview.",
            func=lambda query="": _tool_business_overview(ticker),
        ),
        Tool.from_function(
            name="get_cybersecurity_disclosure",
            description="Get the Cybersecurity risk management and governance disclosure (10-K Item 1C, SEC-mandated since 2023).",
            func=lambda query="": _tool_cybersecurity_disclosure(ticker),
        ),
        Tool.from_function(
            name="get_legal_proceedings",
            description="Get Legal Proceedings section (10-K Item 3): significant pending litigation, regulatory actions, and legal risks.",
            func=lambda query="": _tool_legal_proceedings(ticker),
        ),
        Tool.from_function(
            name="get_all_summaries",
            description="Get a comprehensive overview across risks, MD&A, and financials.",
            func=None,
            coroutine=lambda query="": _tool_all_summaries(ticker, llm),
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
            func=None,
            coroutine=lambda query="": _tool_analyze_latest_8k(ticker, llm),
        ),
        Tool.from_function(
            name="get_8k_item",
            description=(
                "Get raw text of a specific 8-K item by number (e.g. '2.02' for earnings, "
                "'1.01' for material agreements, '5.02' for leadership changes). "
                "Pass the item number as the query."
            ),
            func=lambda item="2.02": _tool_8k_item(ticker, item),
        ),
    ]

    return tools, llm_id
