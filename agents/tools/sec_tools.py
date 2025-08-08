from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any, Optional

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from agents.sec_workflow.sec_llm_models import SECDocumentProcessor


# Global cache for shared data retrievers and processors
_shared_retrievers: Dict[str, SECDataRetrieval] = {}
_shared_processors: Dict[str, SECDocumentProcessor] = {}
_processed_cache: Dict[str, Dict[str, Any]] = {}

# Current context (set once per session/ticker)
_current_ticker: Optional[str] = None
_current_llm_id: Optional[str] = None


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


def set_sec_context(ticker: str, llm: BaseChatModel) -> str:
    """Set the current SEC analysis context (ticker and LLM).

    Returns a context id string for reference.
    """
    global _current_ticker, _current_llm_id
    _current_ticker = ticker
    _current_llm_id = f"{ticker}_{id(llm)}"
    # Ensure shared instances exist (single SEC API call per ticker)
    _get_shared_retriever(ticker)
    _get_shared_processor(ticker, llm)
    return _current_llm_id


def _ensure_context() -> Optional[str]:
    if not _current_ticker:
        return "SEC context not initialized."
    return None


# Internal implementation helpers (no decorators)
def _impl_get_raw_risk_factors() -> str:
    err = _ensure_context()
    if err:
        return err
    try:
        retriever = _get_shared_retriever(_current_ticker)  # type: ignore[arg-type]
        return retriever.extract_risk_factors()
    except Exception as e:
        return f"Failed to retrieve risk factors for {_current_ticker}: {e}"


def _impl_get_risk_factors_summary() -> str:
    err = _ensure_context()
    if err:
        return err
    ticker = _current_ticker  # type: ignore[assignment]
    cache_key = _get_cache_key(ticker, "risk_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _shared_processors.get(_current_llm_id)  # type: ignore[arg-type]
            if not processor:
                return "SEC processor not initialized."
            risk_text = retriever.extract_risk_factors()
            analysis = processor.analyze_risk_factors(ticker, risk_text)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze risk factors for {ticker}: {e}"
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        sentiment = result.get("sentiment_score", "N/A")
        key_risks = result.get("key_risks", [])[:3]
        summary = f"Risk Sentiment Score: {sentiment}/10\n\nTop Risk Factors:\n"
        for i, risk in enumerate(key_risks, 1):
            summary += f"{i}. {risk}\n"
        return summary
    return str(result)


def _impl_get_raw_mda() -> str:
    err = _ensure_context()
    if err:
        return err
    try:
        retriever = _get_shared_retriever(_current_ticker)  # type: ignore[arg-type]
        return retriever.extract_management_discussion()
    except Exception as e:
        return f"Failed to retrieve MD&A for {_current_ticker}: {e}"


def _impl_get_mda_summary() -> str:
    err = _ensure_context()
    if err:
        return err
    ticker = _current_ticker  # type: ignore[assignment]
    cache_key = _get_cache_key(ticker, "mda_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _shared_processors.get(_current_llm_id)  # type: ignore[arg-type]
            if not processor:
                return "SEC processor not initialized."
            mda_text = retriever.extract_management_discussion()
            analysis = processor.analyze_mda(ticker, mda_text)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze MD&A for {ticker}: {e}"
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        sentiment = result.get("sentiment_score", "N/A")
        outlook = result.get("future_outlook", "N/A")
        key_points = result.get("key_points", [])[:3]
        summary = (
            f"Management Sentiment: {sentiment}\n\n"
            f"Future Outlook: {outlook}\n\nKey Points:\n"
        )
        for i, point in enumerate(key_points, 1):
            summary += f"{i}. {point}\n"
        return summary
    return str(result)


def _impl_get_raw_balance_sheets() -> str:
    err = _ensure_context()
    if err:
        return err
    try:
        retriever = _get_shared_retriever(_current_ticker)  # type: ignore[arg-type]
        result = retriever.extract_balance_sheet_as_json()
        if isinstance(result, dict) and "error" not in result:
            return f"Balance sheet data available for: {list(result.keys())}"
        return str(result)
    except Exception as e:
        return f"Failed to retrieve balance sheets for {_current_ticker}: {e}"


def _impl_get_balance_sheet_summary() -> str:
    err = _ensure_context()
    if err:
        return err
    ticker = _current_ticker  # type: ignore[assignment]
    cache_key = _get_cache_key(ticker, "balance_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _shared_processors.get(_current_llm_id)  # type: ignore[arg-type]
            if not processor:
                return "SEC processor not initialized."
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
        key_metrics = result.get("key_metrics", [])[:3]
        red_flags = result.get("red_flags", [])
        response = f"Financial Summary: {summary}\n\nKey Metrics:\n"
        for i, metric in enumerate(key_metrics, 1):
            response += f"{i}. {metric}\n"
        if red_flags:
            response += f"\nRed Flags: {', '.join(red_flags[:2])}"
        return response
    return str(result)


def _impl_get_complete_10k_text() -> str:
    err = _ensure_context()
    if err:
        return err
    try:
        retriever = _get_shared_retriever(_current_ticker)  # type: ignore[arg-type]
        result = {
            "ticker": _current_ticker,
            "management_discussion": retriever.extract_management_discussion(),
            "risk_factors": retriever.extract_risk_factors(),
            "filing_type": "10-K comprehensive text",
        }
        sections = [key for key in result.keys() if key != "ticker"]
        return f"Complete 10-K sections available: {', '.join(sections)}"
    except Exception as e:
        return f"Failed to retrieve complete 10-K for {_current_ticker}: {e}"


def _impl_get_all_summaries() -> str:
    err = _ensure_context()
    if err:
        return err
    risk_summary = _impl_get_risk_factors_summary()
    mda_summary = _impl_get_mda_summary()
    balance_summary = _impl_get_balance_sheet_summary()
    summary = f"Comprehensive Analysis for {_current_ticker}:\n\n"
    summary += f"=== RISK ANALYSIS ===\n{risk_summary}\n\n"
    summary += f"=== MANAGEMENT OUTLOOK ===\n{mda_summary}\n\n"
    summary += f"=== FINANCIAL HEALTH ===\n{balance_summary}\n"
    return summary


# Exposed tool functions (no nested defs, no per-call params)
@tool
def get_raw_risk_factors_tool(query: str = "") -> str:
    """Get the complete raw text of Risk Factors section from 10-K filing."""
    return _impl_get_raw_risk_factors()


@tool
def get_risk_factors_summary_tool(query: str = "") -> str:
    """Get structured analysis of Risk Factors with sentiment and key risks."""
    return _impl_get_risk_factors_summary()


@tool
def get_raw_management_discussion_tool(query: str = "") -> str:
    """Get raw text of Management Discussion & Analysis (MD&A)."""
    return _impl_get_raw_mda()


@tool
def get_mda_summary_tool(query: str = "") -> str:
    """Get structured MD&A summary with sentiment, outlook, and key points."""
    return _impl_get_mda_summary()


@tool
def get_raw_balance_sheets_tool(query: str = "") -> str:
    """Get availability of raw balance sheet JSON data (10-K and 10-Q)."""
    return _impl_get_raw_balance_sheets()


@tool
def get_balance_sheet_summary_tool(query: str = "") -> str:
    """Get balance sheet summary with key metrics and red flags."""
    return _impl_get_balance_sheet_summary()


@tool
def get_complete_10k_text_tool(query: str = "") -> str:
    """Get list of available 10-K sections retrieved as raw text."""
    return _impl_get_complete_10k_text()


@tool
def get_all_summaries_tool(query: str = "") -> str:
    """Get a comprehensive overview across risks, MD&A, and financials."""
    return _impl_get_all_summaries()


def create_sec_tools(ticker: str, llm: BaseChatModel) -> tuple[list, str]:
    """Set context and return the list of SEC tools (no nested defs)."""
    llm_id = set_sec_context(ticker, llm)
    tools = [
        get_raw_risk_factors_tool,
        get_risk_factors_summary_tool,
        get_raw_management_discussion_tool,
        get_mda_summary_tool,
        get_raw_balance_sheets_tool,
        get_balance_sheet_summary_tool,
        get_complete_10k_text_tool,
        get_all_summaries_tool,
    ]
    return tools, llm_id
