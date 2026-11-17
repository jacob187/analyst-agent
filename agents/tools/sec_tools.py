import asyncio
import json
from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Callable, Dict, Any

from agents.sec_workflow.analysis_cache import (
    CachedAnalysis,
    FilingRef,
    eightk_kind,
    eightk_overview_and_input,
    get_or_analyze,
    get_retriever,
    latest_filing,
    latest_filings,
    run_analysis,
)
from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from agents.tools.tool_args import DEFAULT_8K_ITEM, EightKItemArgs, NoArgs


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


def _no_10k(ticker: str) -> ValueError:
    return ValueError(
        f"No 10-K filing found for {ticker}. "
        f"This company may file different forms (e.g., N-CSR for investment trusts)."
    )


async def _fresh_refs(ticker: str, forms: tuple[str, ...]) -> Dict[str, FilingRef | None]:
    """The latest ``forms`` filings, after the shared freshness check that
    points the retriever at them. Every SEC tool reads through it, so the
    steps of one plan running in parallel read the same filings. ValueError
    when a 10-K is asked for and the company files none (SPACs, investment
    trusts, foreign filers)."""
    refs = await latest_filings(ticker, forms)
    if "10-K" in refs and refs["10-K"] is None:
        raise _no_10k(ticker)
    return refs


async def _fresh_retriever(ticker: str, *forms: str) -> SECDataRetrieval:
    """The shared retriever, synced with EDGAR's latest ``forms`` filings."""
    await _fresh_refs(ticker, forms)
    return await asyncio.to_thread(get_retriever, ticker)


def _read_ref(form: str, read: Dict[str, Any], looked_up: FilingRef) -> FilingRef:
    """The filing ``read`` (a section / overview result) came from: the key
    its analysis is saved under. A concurrent refresh can move the shared
    retriever to a newer filing between the latest-filing lookup and the
    read, so the looked-up ref is only the fallback."""
    meta = read.get("metadata") or {}
    if not meta.get("accession"):
        return looked_up
    return FilingRef(form, meta["accession"], meta.get("filing_date", ""))


def _served(result: CachedAnalysis) -> str:
    return _dump_analysis_json(result.served())


async def _serve_analysis(
    ticker: str,
    llm: BaseChatModel,
    filing: FilingRef,
    analysis_type: str,
    load_input: Callable[[], tuple[FilingRef, Dict[str, Any]]],
) -> str:
    """The analysis of ``filing`` as JSON with its ``filing_reference``.

    A stored analysis (``filing_analyses``, shared with the company page) is
    served without downloading the filing. Otherwise ``load_input()``
    (blocking edgartools, run in a thread) returns the input and the filing
    it was read from, and the analysis is generated and saved under that
    filing. The LLM runs via ``ainvoke`` in this task's context, so the chat
    run's callbacks (tracing, LLMCallCharger) see the call."""
    hit = await get_or_analyze(ticker, filing, analysis_type)
    if hit is not None:
        return _served(hit)
    read, raw = await asyncio.to_thread(load_input)

    async def generate() -> Dict[str, Any]:
        return await run_analysis(llm, ticker, analysis_type, raw)

    return _served(await get_or_analyze(ticker, read, analysis_type, generate))


async def _best_section_summary(
    ticker: str, llm: BaseChatModel, getter: str, stem: str,
) -> str | None:
    """Analysis of the 10-Q section when substantive, else the 10-K's, keyed
    like the company page (``{stem}_10q`` / ``{stem}_10k``). None when the
    section is missing from both.

    Whether the 10-Q text is substantive is only known once it is read (the
    company page stores ``{stem}_10q`` even for boilerplate), so the 10-Q is
    read first; a stored 10-K analysis is then served without reading the 10-K.
    """
    refs = await _fresh_refs(ticker, ("10-K", "10-Q"))
    tenk, tenq = refs["10-K"], refs["10-Q"]
    retriever = await asyncio.to_thread(get_retriever, ticker)
    fetch = getattr(retriever, getter)
    if tenq is not None:
        section = await asyncio.to_thread(fetch, "10-Q")
        if section.get("found") and _is_substantive(section.get("text", "")):
            read = _read_ref("10-Q", section, tenq)
            return await _serve_analysis(
                ticker, llm, read, f"{stem}_10q", lambda: (read, section),
            )

    def load_10k() -> tuple[FilingRef, Dict[str, Any]]:
        section = fetch("10-K")
        if not section.get("found"):
            raise _SectionUnavailable(section.get("text", ""))
        return _read_ref("10-K", section, tenk), section

    try:
        return await _serve_analysis(ticker, llm, tenk, f"{stem}_10k", load_10k)
    except _SectionUnavailable:
        return None


async def _tool_raw_risk_factors(ticker: str, llm: BaseChatModel) -> str:
    """Return raw Risk Factors text, preferring 10-Q (most recent) with 10-K fallback."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K", "10-Q")
        result = await asyncio.to_thread(_fetch_best_section, retriever.get_risk_factors_raw)
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


async def _tool_raw_mda(ticker: str, llm: BaseChatModel) -> str:
    """Return raw MD&A text, preferring 10-Q (most recent) with 10-K fallback."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K", "10-Q")
        result = await asyncio.to_thread(_fetch_best_section, retriever.get_mda_raw)
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


async def _tool_raw_balance_sheets(ticker: str, llm: BaseChatModel) -> str:
    """Return availability list of balance sheet JSON sections for the given ticker."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K", "10-Q")
        result = await asyncio.to_thread(retriever.extract_balance_sheet_as_json)
        if isinstance(result, dict) and "error" not in result:
            return f"Balance sheet data available for: {list(result.keys())}"
        return str(result)
    except Exception as e:
        return f"Failed to retrieve balance sheets for {ticker}: {e}"


async def _tool_balance_sheet_summary(ticker: str, llm: BaseChatModel) -> str:
    """Analysis of the latest balance sheet (cached): the 10-Q's when the
    10-Q is newer than the 10-K and has one, else the 10-K's.

    Each is keyed by the filing it was read from. The 10-K's shares the
    company page's ``balance`` key (the same 10-K-only input); the 10-Q's is
    ``balance_10q`` under the 10-Q, so it never overwrites the page's.
    """

    def load(form: str, key: str) -> Callable[[], tuple[FilingRef, Dict[str, Any]]]:
        def load_input() -> tuple[FilingRef, Dict[str, Any]]:
            try:
                text, meta = get_retriever(ticker).balance_sheet_text(form)
            except Exception as e:
                raise _SectionUnavailable(f"No {form} balance sheet: {e}") from e
            return FilingRef(form, meta.accession, meta.filing_date), {key: text}

        return load_input

    try:
        refs = await _fresh_refs(ticker, ("10-K", "10-Q"))
        tenk, tenq = refs["10-K"], refs["10-Q"]
        if tenq is not None and tenq.filing_date > tenk.filing_date:
            try:
                return await _serve_analysis(ticker, llm, tenq, "balance_10q", load("10-Q", "tenq"))
            except _SectionUnavailable:
                pass  # e.g. no XBRL balance sheet: fall back to the 10-K's
        return await _serve_analysis(ticker, llm, tenk, "balance", load("10-K", "tenk"))
    except Exception as e:
        return f"Failed to analyze balance sheet for {ticker}: {e}"


async def _tool_business_overview(ticker: str) -> str:
    """Return raw Business Overview text (10-K Item 1)."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K")
        result = await asyncio.to_thread(retriever.get_business_raw)
        if not result.get("found"):
            return f"Business overview not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve business overview for {ticker}: {e}"


async def _tool_cybersecurity_disclosure(ticker: str) -> str:
    """Return Cybersecurity risk management disclosure (10-K Item 1C)."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K")
        result = await asyncio.to_thread(retriever.get_cybersecurity_raw)
        if not result.get("found"):
            return f"Cybersecurity disclosure not found for {ticker} (may not be present in older filings)."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve cybersecurity disclosure for {ticker}: {e}"


async def _tool_legal_proceedings(ticker: str) -> str:
    """Return Legal Proceedings text (10-K Item 3)."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K")
        result = await asyncio.to_thread(retriever.get_legal_proceedings_raw)
        if not result.get("found"):
            return f"Legal proceedings section not found for {ticker}."
        return result["text"]
    except Exception as e:
        return f"Failed to retrieve legal proceedings for {ticker}: {e}"


async def _tool_complete_10k_text(ticker: str, llm: BaseChatModel) -> str:
    """Return list of available major 10-K sections for the given ticker."""
    try:
        retriever = await _fresh_retriever(ticker, "10-K")
        mda_data = await asyncio.to_thread(retriever.get_mda_raw, "10-K")
        risk_data = await asyncio.to_thread(retriever.get_risk_factors_raw, "10-K")

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


async def _tool_8k_overview(ticker: str) -> str:
    """Return overview of the latest 8-K filing: items, event type, dates."""
    try:
        retriever = await _fresh_retriever(ticker, "8-K")
        result = await asyncio.to_thread(retriever.get_8k_overview)
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


async def _tool_8k_item(ticker: str, item: str) -> str:
    """Return raw text of a specific 8-K item (e.g. '2.02', '1.01')."""
    try:
        retriever = await _fresh_retriever(ticker, "8-K")
        result = await asyncio.to_thread(retriever.get_8k_item, item)
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
        raw: Dict[str, Any] = {}
        if filing is not None:
            # The kind is fixed per filing, so a stored analysis of either
            # kind is this 8-K's: served without loading the filing.
            hits = [
                hit for kind in ("earnings", "event")
                if (hit := await get_or_analyze(ticker, filing, kind)) is not None
            ]
            if len(hits) == 1:
                return _served(hits[0])
            retriever = await asyncio.to_thread(get_retriever, ticker)
            # Kind, cache key and input all from one filing (one read).
            overview, raw = await asyncio.to_thread(eightk_overview_and_input, retriever)
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

    kind = eightk_kind(overview)
    read = _read_ref("8-K", overview, filing)

    def load_input() -> tuple[FilingRef, Dict[str, Any]]:
        if kind == "earnings" and raw.get("has_earnings") is False:
            raise _SectionUnavailable(
                f"No earnings data in latest 8-K for {ticker}. "
                f"Reason: {raw.get('reason', 'Unknown')}"
            )
        if kind == "event" and not raw.get("text"):
            raise _SectionUnavailable(
                f"The latest 8-K's item text for {ticker} could not be extracted."
            )
        return read, raw

    try:
        return await _serve_analysis(ticker, llm, read, kind, load_input)
    except _SectionUnavailable as e:
        return json.dumps({"found": False, "reason": str(e)}, indent=2)
    except Exception as e:
        return json.dumps(
            {"error": str(e), "tool": "analyze_latest_8k", "ticker": ticker},
            indent=2,
        )


def create_sec_tools(ticker: str, llm: BaseChatModel) -> list[StructuredTool]:
    """Return the list of SEC tools bound to a specific ticker and LLM.

    Each tool is a plain function (no bound instance methods) wrapped as a
    StructuredTool. Every tool is async-only (``coroutine=``), so invoke it
    with ``ainvoke``: each one awaits the shared latest-filing check before
    reading, and the LLM analyses also read and write the DB.

    Args:
        ticker: Company ticker symbol
        llm: LLM for document analysis
    """
    tools: list[StructuredTool] = [
        StructuredTool.from_function(
            name="get_raw_risk_factors",
            description="Get the complete raw text of Risk Factors section from 10-K filing.",
            func=None,
            coroutine=lambda: _tool_raw_risk_factors(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_risk_factors_summary",
            description="Get structured analysis of Risk Factors with sentiment and key risks.",
            func=None,
            coroutine=lambda: _tool_risk_factors_summary(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_raw_management_discussion",
            description="Get raw text of Management Discussion & Analysis (MD&A).",
            func=None,
            coroutine=lambda: _tool_raw_mda(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_mda_summary",
            description="Get structured MD&A summary with sentiment, outlook, and key points.",
            func=None,
            coroutine=lambda: _tool_mda_summary(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_raw_balance_sheets",
            description="Get availability of raw balance sheet JSON data (10-K and 10-Q).",
            func=None,
            coroutine=lambda: _tool_raw_balance_sheets(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_balance_sheet_summary",
            description="Get balance sheet summary with key metrics and red flags.",
            func=None,
            coroutine=lambda: _tool_balance_sheet_summary(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_complete_10k_text",
            description="Get list of available 10-K sections retrieved as raw text.",
            func=None,
            coroutine=lambda: _tool_complete_10k_text(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_business_overview",
            description="Get the Company Business section (10-K Item 1): products, services, segments, and market overview.",
            func=None,
            coroutine=lambda: _tool_business_overview(ticker),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_cybersecurity_disclosure",
            description="Get the Cybersecurity risk management and governance disclosure (10-K Item 1C, SEC-mandated since 2023).",
            func=None,
            coroutine=lambda: _tool_cybersecurity_disclosure(ticker),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_legal_proceedings",
            description="Get Legal Proceedings section (10-K Item 3): significant pending litigation, regulatory actions, and legal risks.",
            func=None,
            coroutine=lambda: _tool_legal_proceedings(ticker),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_all_summaries",
            description="Get a comprehensive overview across risks, MD&A, and financials.",
            func=None,
            coroutine=lambda: _tool_all_summaries(ticker, llm),
            args_schema=NoArgs,
        ),
        # ── 8-K tools ────────────────────────────────────────────────────
        # Dispatcher pattern: a single planner-visible tool routes internally
        # to either the earnings analyzer or the material event analyzer.
        # Replaces the older overview / earnings_summary / material_event_summary
        # trio, which let the planner pick overlapping tools that classified
        # the same filing differently.
        StructuredTool.from_function(
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
            coroutine=lambda: _tool_analyze_latest_8k(ticker, llm),
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            name="get_8k_item",
            description=(
                "Get raw text of a specific 8-K item by number (e.g. '2.02' for earnings, "
                "'1.01' for material agreements, '5.02' for leadership changes). "
                "Pass the item number as `item`."
            ),
            func=None,
            # LangChain validates a bare-string call (the plan worker's
            # ``ainvoke("")``) but passes the raw string on, so normalise it
            # through the schema here too: blank means the default item.
            coroutine=lambda item=DEFAULT_8K_ITEM: _tool_8k_item(
                ticker, EightKItemArgs(item=item).item
            ),
            args_schema=EightKItemArgs,
        ),
    ]

    return tools
