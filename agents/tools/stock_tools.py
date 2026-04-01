"""Stock and technical analysis tools.

These tools retrieve stock price data from Yahoo Finance and calculate
technical indicators (RSI, MACD, Bollinger Bands, pattern recognition, etc.).
They are ticker-bound but do not require an LLM or SEC header.
"""

from typing import Dict

from langchain_core.tools import Tool

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


# Per-ticker cache so all stock tools share a single retriever instance
_shared_stock_retrievers: Dict[str, YahooFinanceDataRetrieval] = {}


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


def _tool_financial_metrics(ticker: str) -> str:
    """Return fundamental financial metrics from income statement and balance sheet."""
    try:
        from agents.technical_workflow.process_technical_indicators import TechnicalIndicators

        retriever = _get_shared_stock_retriever(ticker)
        financials = retriever.get_financials()

        if not financials or (not financials.get("income_stmt") and not financials.get("balance_sheet")):
            return f"No financial statement data available for {ticker}. This may be a newly listed company or a company type (e.g., investment trust) that doesn't report standard financials."

        tech = TechnicalIndicators(ticker)
        metrics = tech._calculate_financial_metrics(financials)

        lines = [f"Financial Metrics for {ticker}:"]
        lines.append("=" * 50)

        has_any_metric = False

        if "revenue_growth_yoy" in metrics:
            growth = metrics["revenue_growth_yoy"]
            direction = "+" if growth >= 0 else ""
            lines.append(f"\n📊 REVENUE:")
            lines.append(f"  Year-over-Year Growth: {direction}{growth:.2f}%")
            has_any_metric = True

        if "net_income_growth_yoy" in metrics:
            growth = metrics["net_income_growth_yoy"]
            direction = "+" if growth >= 0 else ""
            lines.append(f"\n💰 NET INCOME:")
            lines.append(f"  Year-over-Year Growth: {direction}{growth:.2f}%")
            has_any_metric = True

        if "debt_to_assets" in metrics:
            lines.append(f"\n📉 LEVERAGE:")
            lines.append(f"  Debt-to-Assets Ratio: {metrics['debt_to_assets']:.2f}%")
            has_any_metric = True

        if not has_any_metric:
            available_info = []
            if financials.get("income_stmt"):
                available_info.append("income statement data exists but lacks standard revenue/income rows")
            if financials.get("balance_sheet"):
                available_info.append("balance sheet data exists but lacks standard asset/liability rows")
            detail = "; ".join(available_info) if available_info else "no recognizable financial data"
            return f"Could not calculate financial metrics for {ticker}: {detail}. This company may use non-standard financial reporting."

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to calculate financial metrics for {ticker}: {e}"


def _tool_advanced_technical_analysis(ticker: str) -> str:
    """Return advanced technical indicators with contextual explanations.

    Calculates ADX, ATR, Stochastic, Volume Profile, and Fibonacci levels,
    then formats each with an interpretation so the LLM agent can explain
    the results without needing to know the formulas.
    """
    try:
        from agents.technical_workflow.process_technical_indicators import TechnicalIndicators

        retriever = _get_shared_stock_retriever(ticker)
        hist = retriever.get_historical_prices(period="1y")
        if hist is None or hist.empty:
            return f"No historical price data available for {ticker}"

        tech = TechnicalIndicators(ticker)
        indicators = tech.calculate_all_indicators(hist)

        lines = [f"Advanced Technical Analysis for {ticker}:"]
        lines.append("=" * 55)

        # ADX
        adx = indicators.get("adx", {})
        if adx:
            lines.append(f"\nADX (Trend Strength):")
            lines.append(f"  ADX: {adx.get('adx', 'N/A')}")
            lines.append(f"  +DI: {adx.get('plus_di', 'N/A')}  -DI: {adx.get('minus_di', 'N/A')}")
            lines.append(f"  Trend Strength: {adx.get('trend_strength', 'N/A').upper()}")
            lines.append(f"  Insight: {adx.get('signal', '')}")

        # ATR
        atr = indicators.get("atr", {})
        if atr:
            lines.append(f"\nATR (Volatility):")
            lines.append(f"  ATR: ${atr.get('atr', 'N/A')}")
            lines.append(f"  ATR %: {atr.get('atr_percent', 'N/A')}%")
            lines.append(f"  Suggested Stop-Loss: ${atr.get('suggested_stop_loss', 'N/A')}")
            lines.append(f"  Volatility Regime: {atr.get('volatility_regime', 'N/A').upper()}")

        # Stochastic
        stoch = indicators.get("stochastic", {})
        if stoch:
            lines.append(f"\nStochastic Oscillator:")
            lines.append(f"  %K: {stoch.get('k_percent', 'N/A')}  %D: {stoch.get('d_percent', 'N/A')}")
            lines.append(f"  Signal: {stoch.get('signal', 'N/A').upper()}")
            lines.append(f"  Crossover: {stoch.get('crossover', 'none')}")

        # Volume Profile
        vp = indicators.get("volume_profile", {})
        if vp:
            lines.append(f"\nVolume Profile:")
            lines.append(f"  Point of Control (POC): ${vp.get('poc', 'N/A')}")
            lines.append(f"  Value Area: ${vp.get('value_area_low', 'N/A')} - ${vp.get('value_area_high', 'N/A')}")
            lines.append(f"  Price Position: {vp.get('position', 'N/A').replace('_', ' ').upper()}")

        # Fibonacci
        fib = indicators.get("fibonacci", {})
        if fib:
            lines.append(f"\nFibonacci Retracement:")
            lines.append(f"  Swing High: ${fib.get('swing_high', 'N/A')}  Swing Low: ${fib.get('swing_low', 'N/A')}")
            lvls = fib.get("levels", {})
            for label, price in lvls.items():
                lines.append(f"  {label}: ${price}")
            lines.append(f"  Closest Level: {fib.get('closest_level', 'N/A')} (${fib.get('closest_price', 'N/A')}, {fib.get('distance_to_closest', 'N/A')}% away)")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to calculate advanced technical indicators for {ticker}: {e}"


def _tool_detect_patterns(ticker: str) -> str:
    """Detect chart patterns and provide analysis."""
    try:
        from agents.technical_workflow.pattern_recognition import PatternRecognitionEngine

        retriever = _get_shared_stock_retriever(ticker)
        hist = retriever.get_historical_prices(period="1y")
        if hist is None or hist.empty:
            return f"No historical price data available for {ticker}"

        engine = PatternRecognitionEngine()
        patterns = engine.detect_all_patterns(hist)

        if not patterns:
            return f"No significant chart patterns detected for {ticker} in the current data."

        lines = [f"Chart Pattern Detection for {ticker}:"]
        lines.append("=" * 55)

        for p in patterns:
            lines.append(f"\n{p['type'].replace('_', ' ').title()}:")
            lines.append(f"  Direction: {p.get('direction', 'N/A').upper()}")
            lines.append(f"  Confidence: {p.get('confidence', 0):.0%}")
            lines.append(f"  Status: {p.get('status', 'N/A')}")
            if "target" in p:
                lines.append(f"  Price Target: ${p['target']:.2f}")
            if "neckline" in p:
                lines.append(f"  Neckline: ${p['neckline']:.2f}")
            if "cross_date" in p:
                lines.append(f"  Cross Date: {p['cross_date']}")
            if "price_low" in p:
                lines.append(f"  Price Low: ${p['price_low']:.2f}")
            if "price_high" in p:
                lines.append(f"  Price High: ${p['price_high']:.2f}")
            if "rsi_low" in p:
                lines.append(f"  RSI Low: {p['rsi_low']:.1f}")
            if "rsi_high" in p:
                lines.append(f"  RSI High: {p['rsi_high']:.1f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to detect patterns for {ticker}: {e}"


def _tool_multi_timeframe_analysis(ticker: str) -> str:
    """Analyze stock across multiple timeframes."""
    try:
        from agents.technical_workflow.multi_timeframe import MultiTimeframeAnalyzer

        analyzer = MultiTimeframeAnalyzer(ticker)
        result = analyzer.analyze_all_timeframes()

        lines = [f"Multi-Timeframe Analysis for {ticker}:"]
        lines.append("=" * 55)

        for tf_name, tf_data in result.get("timeframes", {}).items():
            if "error" in tf_data:
                lines.append(f"\n{tf_name.upper()}: {tf_data['error']}")
                continue
            trend = tf_data.get("trend", "unknown")
            bars = tf_data.get("bars", 0)
            lines.append(f"\n{tf_name.upper()} ({bars} bars):")
            lines.append(f"  Trend: {trend.upper()}")

            ind = tf_data.get("indicators", {})
            rsi = ind.get("rsi", {})
            if rsi:
                lines.append(f"  RSI: {rsi.get('current', 'N/A'):.1f} ({rsi.get('signal', 'N/A')})")
            macd = ind.get("macd", {})
            if macd:
                lines.append(f"  MACD Signal: {macd.get('signal', 'N/A').upper()}")

        conflicts = result.get("conflicts", [])
        if conflicts:
            lines.append("\nCONFLICTS DETECTED:")
            for c in conflicts:
                lines.append(f"  - [{c['type'].upper()}] {c['detail']}")

        rec = result.get("recommendation", {})
        if rec:
            lines.append(f"\nRECOMMENDATION:")
            lines.append(f"  Bias: {rec.get('bias', 'N/A').upper()}")
            lines.append(f"  Confidence: {rec.get('confidence', 0):.0%}")
            lines.append(f"  Strategy: {rec.get('strategy', 'N/A')}")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to perform multi-timeframe analysis for {ticker}: {e}"


def create_stock_tools(ticker: str) -> list[Tool]:
    """Return stock/technical analysis tools bound to a specific ticker."""
    return [
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
            description="Get current stock info including current price, trailing P/E ratio, forward P/E ratio, price-to-book ratio, market cap, 52-week high/low, volume, average volume, dividend yield, and beta.",
            func=lambda query="": _tool_stock_info(ticker),
        ),
        Tool.from_function(
            name="get_financial_metrics",
            description="Get fundamental financial metrics including year-over-year revenue growth, net income growth, and debt-to-assets ratio from income statement and balance sheet.",
            func=lambda query="": _tool_financial_metrics(ticker),
        ),
        Tool.from_function(
            name="get_advanced_technical_analysis",
            description="Get advanced technical indicators: ADX (trend strength), ATR (volatility/position sizing), Stochastic (momentum), Volume Profile (institutional levels), and Fibonacci retracement levels.",
            func=lambda query="": _tool_advanced_technical_analysis(ticker),
        ),
        Tool.from_function(
            name="get_pattern_detection",
            description="Detect chart patterns: Head & Shoulders, Double Top/Bottom, Golden/Death Cross, RSI divergences. Returns pattern type, confidence, direction, and price targets.",
            func=lambda query="": _tool_detect_patterns(ticker),
        ),
        Tool.from_function(
            name="get_multi_timeframe_analysis",
            description="Analyze stock across daily, weekly, and hourly timeframes. Detects conflicts between timeframes and provides weighted trading recommendation.",
            func=lambda query="": _tool_multi_timeframe_analysis(ticker),
        ),
    ]
