from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from agents.sec_workflow.sec_llm_models import SECDocumentProcessor


# Global cache for shared data retrievers, processors, and processed outputs
_shared_retrievers: Dict[str, SECDataRetrieval] = {}
_shared_processors: Dict[str, SECDocumentProcessor] = {}
_processed_cache: Dict[str, Dict[str, Any]] = {}


def _get_shared_retriever(ticker: str) -> SECDataRetrieval:
    """Get or create a shared SEC data retriever for the ticker."""
    if ticker not in _shared_retrievers:
        print(f"Making single SEC API call for {ticker}...")
        _shared_retrievers[ticker] = SECDataRetrieval(ticker)
        _processed_cache[ticker] = {}
    return _shared_retrievers[ticker]


def _get_shared_processor(ticker: str, llm: BaseChatModel) -> SECDocumentProcessor:
    """Get or create a shared SEC document processor for the ticker."""
    processor_key = f"{ticker}_{id(llm)}"
    if processor_key not in _shared_processors:
        _shared_processors[processor_key] = SECDocumentProcessor(llm)
    return _shared_processors[processor_key]


def _get_cache_key(ticker: str, analysis_type: str) -> str:
    """Generate cache key for processed analysis."""
    return f"{ticker}_{analysis_type}"


def _format_top_n(items: list[str], n: int = 3) -> str:
    lines = []
    for i, item in enumerate(items[:n], 1):
        lines.append(f"{i}. {item}")
    return "\n".join(lines)


def _tool_raw_risk_factors(ticker: str, llm: BaseChatModel) -> str:
    """Return raw Risk Factors text for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker)
        return retriever.extract_risk_factors()
    except Exception as e:
        return f"Failed to retrieve risk factors for {ticker}: {e}"


def _tool_risk_factors_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized Risk Factors for the given ticker (cached)."""
    cache_key = _get_cache_key(ticker, "risk_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _get_shared_processor(ticker, llm)
            risk_text = retriever.extract_risk_factors()
            analysis = processor.analyze_risk_factors(ticker, risk_text)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze risk factors for {ticker}: {e}"
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        sentiment = result.get("sentiment_score", "N/A")
        key_risks = result.get("key_risks", [])
        summary = (
            f"Risk Sentiment Score: {sentiment}/10\n\nTop Risk Factors:\n"
            f"{_format_top_n(key_risks, 3)}"
        )
        return summary
    return str(result)


def _tool_raw_mda(ticker: str, llm: BaseChatModel) -> str:
    """Return raw MD&A text for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker)
        return retriever.extract_management_discussion()
    except Exception as e:
        return f"Failed to retrieve MD&A for {ticker}: {e}"


def _tool_mda_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized MD&A for the given ticker (cached)."""
    cache_key = _get_cache_key(ticker, "mda_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _get_shared_processor(ticker, llm)
            mda_text = retriever.extract_management_discussion()
            analysis = processor.analyze_mda(ticker, mda_text)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze MD&A for {ticker}: {e}"
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        sentiment = result.get("sentiment_score", "N/A")
        outlook = result.get("future_outlook", "N/A")
        key_points = result.get("key_points", [])
        summary = (
            f"Management Sentiment: {sentiment}\n\n"
            f"Future Outlook: {outlook}\n\nKey Points:\n{_format_top_n(key_points, 3)}"
        )
        return summary
    return str(result)


def _tool_raw_balance_sheets(ticker: str, llm: BaseChatModel) -> str:
    """Return availability list of balance sheet JSON sections for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker)
        result = retriever.extract_balance_sheet_as_json()
        if isinstance(result, dict) and "error" not in result:
            return f"Balance sheet data available for: {list(result.keys())}"
        return str(result)
    except Exception as e:
        return f"Failed to retrieve balance sheets for {ticker}: {e}"


def _tool_balance_sheet_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized balance sheet analysis for the given ticker (cached)."""
    cache_key = _get_cache_key(ticker, "balance_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
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
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        summary = result.get("summary", "N/A")
        key_metrics = result.get("key_metrics", [])
        red_flags = result.get("red_flags", [])
        response = f"Financial Summary: {summary}\n\nKey Metrics:\n{_format_top_n(key_metrics, 3)}"
        if red_flags:
            response += f"\n\nRed Flags: {', '.join(red_flags[:2])}"
        return response
    return str(result)


def _tool_complete_10k_text(ticker: str, llm: BaseChatModel) -> str:
    """Return list of available major 10-K sections for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker)
        result = {
            "ticker": ticker,
            "management_discussion": retriever.extract_management_discussion(),
            "risk_factors": retriever.extract_risk_factors(),
            "filing_type": "10-K comprehensive text",
        }
        sections = [key for key in result.keys() if key != "ticker"]
        return f"Complete 10-K sections available: {', '.join(sections)}"
    except Exception as e:
        return f"Failed to retrieve complete 10-K for {ticker}: {e}"


def _tool_all_summaries(ticker: str, llm: BaseChatModel) -> str:
    """Return a comprehensive overview across risks, MD&A, and financials."""
    risk_summary = _tool_risk_factors_summary(ticker, llm)
    mda_summary = _tool_mda_summary(ticker, llm)
    balance_summary = _tool_balance_sheet_summary(ticker, llm)
    summary = f"Comprehensive Analysis for {ticker}:\n\n"
    summary += f"=== RISK ANALYSIS ===\n{risk_summary}\n\n"
    summary += f"=== MANAGEMENT OUTLOOK ===\n{mda_summary}\n\n"
    summary += f"=== FINANCIAL HEALTH ===\n{balance_summary}\n"
    return summary


def create_sec_tools(ticker: str, llm: BaseChatModel) -> tuple[list, str]:
    """Return the list of SEC tools bound to a specific ticker and LLM.

    Each tool is a plain function (no bound instance methods) wrapped as a Tool.
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
            func=lambda query="": _tool_risk_factors_summary(ticker, llm),
        ),
        Tool.from_function(
            name="get_raw_management_discussion",
            description="Get raw text of Management Discussion & Analysis (MD&A).",
            func=lambda query="": _tool_raw_mda(ticker, llm),
        ),
        Tool.from_function(
            name="get_mda_summary",
            description="Get structured MD&A summary with sentiment, outlook, and key points.",
            func=lambda query="": _tool_mda_summary(ticker, llm),
        ),
        Tool.from_function(
            name="get_raw_balance_sheets",
            description="Get availability of raw balance sheet JSON data (10-K and 10-Q).",
            func=lambda query="": _tool_raw_balance_sheets(ticker, llm),
        ),
        Tool.from_function(
            name="get_balance_sheet_summary",
            description="Get balance sheet summary with key metrics and red flags.",
            func=lambda query="": _tool_balance_sheet_summary(ticker, llm),
        ),
        Tool.from_function(
            name="get_complete_10k_text",
            description="Get list of available 10-K sections retrieved as raw text.",
            func=lambda query="": _tool_complete_10k_text(ticker, llm),
        ),
        Tool.from_function(
            name="get_all_summaries",
            description="Get a comprehensive overview across risks, MD&A, and financials.",
            func=lambda query="": _tool_all_summaries(ticker, llm),
        ),
    ]

    return tools, llm_id
