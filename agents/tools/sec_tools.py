from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from agents.sec_workflow.sec_llm_models import SECDocumentProcessor
from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


# Global cache for shared data retrievers, processors, and processed outputs
_shared_retrievers: Dict[str, SECDataRetrieval] = {}
_shared_processors: Dict[str, SECDocumentProcessor] = {}
_shared_stock_retrievers: Dict[str, YahooFinanceDataRetrieval] = {}
_processed_cache: Dict[str, Dict[str, Any]] = {}


def _get_shared_retriever(ticker: str) -> SECDataRetrieval:
    """Get or create a shared SEC data retriever for the ticker.

    Validates that the company has 10-K filings available before proceeding.
    Raises ValueError if no 10-K filing is found.
    """
    if ticker not in _shared_retrievers:
        print(f"Making single SEC API call for {ticker}...")
        retriever = SECDataRetrieval(ticker)

        # Validate that 10-K filings are available
        availability = retriever.check_filing_availability()
        if not availability["has_10k"]:
            raise ValueError(
                f"No 10-K filing found for {availability['company_name']} ({ticker}). "
                f"This company may file different forms (e.g., N-CSR for investment trusts)."
            )

        _shared_retrievers[ticker] = retriever
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
        # Use the new dict-based method that includes metadata
        result = retriever.get_risk_factors_raw("10-K")
        if not result.get("found", False):
            return f"Risk Factors section not found in 10-K filing for {ticker}."
        return result.get("text", "No text available")
    except Exception as e:
        return f"Failed to retrieve risk factors for {ticker}: {e}"


def _tool_risk_factors_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized Risk Factors for the given ticker (cached)."""
    cache_key = _get_cache_key(ticker, "risk_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _get_shared_processor(ticker, llm)
            # Use the new dict-based method that includes metadata
            risk_data = retriever.get_risk_factors_raw("10-K")
            if not risk_data.get("found", False):
                return f"Risk Factors section not found in 10-K filing for {ticker}."
            # Pass the full dict to the analysis method
            analysis = processor.analyze_risk_factors(ticker, risk_data)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze risk factors for {ticker}: {e}"
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        sentiment = result.get("sentiment_score", "N/A")
        key_risks = result.get("key_risks", [])
        summary = result.get("summary", "")
        response = f"Risk Analysis Summary:\n{summary}\n\nRisk Severity Score: {sentiment}/10\n\nTop Risk Factors:\n{_format_top_n(key_risks, 3)}"
        return response
    return str(result)


def _tool_raw_mda(ticker: str, llm: BaseChatModel) -> str:
    """Return raw MD&A text for the given ticker."""
    try:
        retriever = _get_shared_retriever(ticker)
        # Use the new dict-based method that includes metadata
        result = retriever.get_mda_raw("10-K")
        if not result.get("found", False):
            return f"Management Discussion section not found in 10-K filing for {ticker}."
        return result.get("text", "No text available")
    except Exception as e:
        return f"Failed to retrieve MD&A for {ticker}: {e}"


def _tool_mda_summary(ticker: str, llm: BaseChatModel) -> str:
    """Return summarized MD&A for the given ticker (cached)."""
    cache_key = _get_cache_key(ticker, "mda_summary")
    if cache_key not in _processed_cache.get(ticker, {}):
        try:
            retriever = _get_shared_retriever(ticker)
            processor = _get_shared_processor(ticker, llm)
            # Use the new dict-based method that includes metadata
            mda_data = retriever.get_mda_raw("10-K")
            if not mda_data.get("found", False):
                return f"Management Discussion section not found in 10-K filing for {ticker}."
            # Pass the full dict to the analysis method
            analysis = processor.analyze_mda(ticker, mda_data)
            _processed_cache[ticker][cache_key] = analysis.model_dump()
        except Exception as e:
            return f"Failed to analyze MD&A for {ticker}: {e}"
    result = _processed_cache[ticker][cache_key]
    if isinstance(result, dict) and "error" not in result:
        sentiment = result.get("sentiment_score", "N/A")
        outlook = result.get("future_outlook", "N/A")
        key_points = result.get("key_points", [])
        summary_text = result.get("summary", "")
        response = (
            f"MD&A Summary:\n{summary_text}\n\n"
            f"Management Sentiment: {sentiment}\n\n"
            f"Future Outlook: {outlook}\n\nKey Points:\n{_format_top_n(key_points, 3)}"
        )
        return response
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


def _get_shared_stock_retriever(ticker: str) -> YahooFinanceDataRetrieval:
    """Get or create a shared Yahoo Finance data retriever for the ticker."""
    if ticker not in _shared_stock_retrievers:
        _shared_stock_retrievers[ticker] = YahooFinanceDataRetrieval(ticker)
    return _shared_stock_retrievers[ticker]


def _tool_stock_price_history(ticker: str, period: str = "1mo") -> str:
    """Return recent stock price history for the given ticker."""
    try:
        retriever = _get_shared_stock_retriever(ticker)
        hist = retriever.get_historical_prices(period=period)
        if hist is None or hist.empty:
            return f"No historical price data available for {ticker}"

        # Get last 10 trading days
        recent = hist.tail(10)
        lines = [f"Stock Price History for {ticker} (last {len(recent)} trading days):"]
        lines.append("-" * 50)

        for date, row in recent.iterrows():
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]
            lines.append(f"{date_str}: Open ${row['Open']:.2f}, Close ${row['Close']:.2f}, High ${row['High']:.2f}, Low ${row['Low']:.2f}")

        # Add summary
        if len(hist) >= 2:
            start_price = hist.iloc[0]['Close']
            end_price = hist.iloc[-1]['Close']
            change = ((end_price - start_price) / start_price) * 100
            lines.append("-" * 50)
            lines.append(f"Period change: {change:+.2f}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to retrieve stock prices for {ticker}: {e}"


def _tool_technical_analysis(ticker: str) -> str:
    """Return technical indicators including RSI, MACD, Bollinger Bands, and moving averages."""
    try:
        from agents.technical_workflow.process_technical_indicators import TechnicalIndicators

        retriever = _get_shared_stock_retriever(ticker)
        hist = retriever.get_historical_prices(period="1y")
        if hist is None or hist.empty:
            return f"No historical price data available for {ticker}"

        tech = TechnicalIndicators(ticker)
        indicators = tech.calculate_all_indicators(hist)

        lines = [f"Technical Analysis for {ticker}:"]
        lines.append("=" * 50)

        # Current price and moving averages
        if "moving_averages" in indicators:
            ma = indicators["moving_averages"]
            lines.append("\n📈 PRICE & MOVING AVERAGES:")
            lines.append(f"  Current Price: ${ma.get('latest_close', 0):.2f}")
            lines.append(f"  5-day MA: ${ma.get('MA_5', 0):.2f}")
            lines.append(f"  10-day MA: ${ma.get('MA_10', 0):.2f}")
            lines.append(f"  20-day MA: ${ma.get('MA_20', 0):.2f}")
            if "MA_50" in ma:
                lines.append(f"  50-day MA: ${ma['MA_50']:.2f}")
            if "MA_200" in ma:
                lines.append(f"  200-day MA: ${ma['MA_200']:.2f}")
            if "trend_50_200" in ma:
                trend = ma["trend_50_200"]
                lines.append(f"  Trend (50/200 MA): {trend.upper()}")

        # RSI
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            lines.append(f"\n📊 RSI (14-day):")
            lines.append(f"  Value: {rsi.get('current', 0):.2f}")
            lines.append(f"  Signal: {rsi.get('signal', 'N/A').upper()}")

        # MACD
        if "macd" in indicators:
            macd = indicators["macd"]
            lines.append(f"\n📉 MACD:")
            lines.append(f"  MACD Line: {macd.get('macd_line', 0):.4f}")
            lines.append(f"  Signal Line: {macd.get('signal_line', 0):.4f}")
            lines.append(f"  Histogram: {macd.get('histogram', 0):.4f}")
            lines.append(f"  Signal: {macd.get('signal', 'N/A').upper()}")

        # Bollinger Bands
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            lines.append(f"\n📏 BOLLINGER BANDS:")
            lines.append(f"  Upper Band: ${bb.get('upper_band', 0):.2f}")
            lines.append(f"  Middle Band: ${bb.get('middle_band', 0):.2f}")
            lines.append(f"  Lower Band: ${bb.get('lower_band', 0):.2f}")
            lines.append(f"  Position: {bb.get('position', 'N/A').replace('_', ' ').upper()}")

        # Volatility
        if "volatility" in indicators:
            vol = indicators["volatility"]
            lines.append(f"\n⚡ VOLATILITY:")
            lines.append(f"  Daily: {vol.get('daily_volatility', 0)*100:.2f}%")
            lines.append(f"  Annualized: {vol.get('annualized_volatility', 0)*100:.2f}%")
            lines.append(f"  Max Drawdown: {vol.get('max_drawdown', 0)*100:.2f}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to calculate technical indicators for {ticker}: {e}"


def _tool_stock_info(ticker: str) -> str:
    """Return current stock info and key metrics for the given ticker."""
    try:
        retriever = _get_shared_stock_retriever(ticker)
        info = retriever.get_info()
        if not info:
            return f"No stock info available for {ticker}"

        lines = [f"Stock Info for {ticker}:"]
        lines.append("-" * 50)

        # Key metrics to display
        key_fields = [
            ("currentPrice", "Current Price"),
            ("previousClose", "Previous Close"),
            ("dayHigh", "Day High"),
            ("dayLow", "Day Low"),
            ("fiftyTwoWeekHigh", "52-Week High"),
            ("fiftyTwoWeekLow", "52-Week Low"),
            ("marketCap", "Market Cap"),
            ("volume", "Volume"),
            ("averageVolume", "Avg Volume"),
            ("trailingPE", "P/E Ratio"),
            ("forwardPE", "Forward P/E"),
            ("priceToBook", "Price/Book"),
            ("dividendYield", "Dividend Yield"),
            ("beta", "Beta"),
        ]

        for field, label in key_fields:
            if field in info and info[field] is not None:
                value = info[field]
                if field == "marketCap":
                    value = f"${value/1e9:.2f}B"
                elif field in ["volume", "averageVolume"]:
                    value = f"{value:,.0f}"
                elif field == "dividendYield":
                    value = f"{value*100:.2f}%"
                elif isinstance(value, float):
                    value = f"${value:.2f}" if "Price" in label or "High" in label or "Low" in label or "Close" in label else f"{value:.2f}"
                lines.append(f"{label}: {value}")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to retrieve stock info for {ticker}: {e}"


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
        Tool.from_function(
            name="get_stock_price_history",
            description="Get recent stock price history (last 10 trading days with open, close, high, low prices).",
            func=lambda query="": _tool_stock_price_history(ticker),
        ),
        Tool.from_function(
            name="get_technical_analysis",
            description="Get technical indicators including RSI, MACD, Bollinger Bands, moving averages (5/10/20/50/200-day), volatility, and trend signals.",
            func=lambda query="": _tool_technical_analysis(ticker),
        ),
        Tool.from_function(
            name="get_stock_info",
            description="Get current stock info including price, P/E ratio, market cap, 52-week high/low, volume, dividend yield, and beta.",
            func=lambda query="": _tool_stock_info(ticker),
        ),
    ]

    return tools, llm_id
