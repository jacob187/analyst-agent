"""Daily briefing service — aggregates technical data and synthesizes via LLM."""

import time
from typing import Any

from agents.prompts import DAILY_BRIEFING_PROMPT


# Cache: (ticker_key, timestamp, text)
_briefing_cache: tuple | None = None
_BRIEFING_CACHE_TTL = 900  # 15 minutes


class BriefingService:
    """Generate a daily market briefing for a watchlist."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, tickers: list[str]) -> str:
        """Build briefing for the given tickers. Uses 15-min cache."""
        global _briefing_cache
        now = time.monotonic()
        cache_key = tuple(sorted(tickers))

        if (_briefing_cache is not None
                and _briefing_cache[0] == cache_key
                and (now - _briefing_cache[1]) < _BRIEFING_CACHE_TTL):
            return _briefing_cache[2]

        ticker_data = self._gather_ticker_data(tickers)
        regime_data = self._get_market_regime()

        briefing = self._synthesize(ticker_data, regime_data)

        _briefing_cache = (cache_key, now, briefing)
        return briefing

    def _gather_ticker_data(self, tickers: list[str]) -> list[dict[str, Any]]:
        """For each ticker: price, RSI, MACD, ADX, patterns."""
        from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
        from agents.technical_workflow.process_technical_indicators import TechnicalIndicators
        from agents.technical_workflow.pattern_recognition import PatternRecognitionEngine

        results = []
        for ticker in tickers:
            entry: dict[str, Any] = {"ticker": ticker}
            try:
                retriever = YahooFinanceDataRetrieval(ticker)
                hist = retriever.get_historical_prices(period="3mo")

                if hist is None or hist.empty:
                    entry["error"] = "No data available"
                    results.append(entry)
                    continue

                # Current price
                entry["price"] = round(float(hist["Close"].iloc[-1]), 2)
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    entry["change_pct"] = round((entry["price"] - prev) / prev * 100, 2)

                # Indicators
                ti = TechnicalIndicators(ticker)
                indicators = ti.calculate_all_indicators(hist)

                if "rsi" in indicators:
                    entry["rsi"] = round(indicators["rsi"].get("current", 0), 1)
                    entry["rsi_signal"] = indicators["rsi"].get("signal", "neutral")

                if "macd" in indicators:
                    entry["macd_signal"] = indicators["macd"].get("signal", "neutral")
                    entry["macd_histogram"] = round(indicators["macd"].get("histogram", 0), 4)

                if "adx" in indicators:
                    entry["adx"] = indicators["adx"].get("adx", 0)
                    entry["trend_strength"] = indicators["adx"].get("trend_strength", "unknown")

                # Patterns
                engine = PatternRecognitionEngine()
                patterns = engine.detect_all_patterns(hist)
                if patterns:
                    entry["patterns"] = [
                        f"{p['type'].replace('_', ' ')} ({p['direction']}, {p['confidence']*100:.0f}%)"
                        for p in patterns[:3]
                    ]

            except Exception as e:
                entry["error"] = str(e)

            results.append(entry)
        return results

    def _get_market_regime(self) -> dict[str, Any]:
        """Get current market regime from SPY/VIX."""
        try:
            from agents.market_analysis.regime_detector import MarketRegimeDetector
            detector = MarketRegimeDetector()
            return detector.detect_regime()
        except Exception:
            return {"error": "Could not determine market regime"}

    def _format_market_context(self, regime: dict[str, Any]) -> str:
        """Format market regime data for the prompt.

        The MarketRegimeDetector returns simple strings for trend/volatility/phase
        (e.g., "bull", "low", "markup"), not nested dicts.
        """
        if "error" in regime:
            return "Market regime data unavailable."

        lines = []
        trend = regime.get("trend", "unknown")
        vol = regime.get("volatility", "unknown")
        phase = regime.get("phase", "unknown")

        lines.append(f"Trend: {str(trend).upper()}")
        lines.append(f"Volatility: {str(vol).upper()}")
        lines.append(f"Phase: {str(phase).upper()}")

        recs = regime.get("recommendations", [])
        if recs:
            lines.append("Strategy: " + "; ".join(recs[:3]))

        return "\n".join(lines)

    def _format_ticker_summary(self, data: list[dict[str, Any]]) -> str:
        """Format ticker data for the prompt."""
        lines = []
        for d in data:
            ticker = d["ticker"]
            if "error" in d:
                lines.append(f"{ticker}: Data unavailable ({d['error']})")
                continue

            parts = [f"{ticker}: ${d.get('price', 0):.2f}"]
            if "change_pct" in d:
                parts.append(f"({d['change_pct']:+.2f}%)")

            if "rsi" in d:
                parts.append(f"RSI={d['rsi']:.1f} [{d.get('rsi_signal', '').upper()}]")
            if "macd_signal" in d:
                parts.append(f"MACD={d.get('macd_signal', '').upper()}")
            if "adx" in d:
                parts.append(f"ADX={d['adx']:.1f} [{d.get('trend_strength', '').upper()}]")
            if "patterns" in d:
                parts.append(f"Patterns: {', '.join(d['patterns'])}")

            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _synthesize(self, ticker_data: list[dict[str, Any]], regime_data: dict[str, Any]) -> str:
        """Format prompt and invoke LLM."""
        market_context = self._format_market_context(regime_data)
        ticker_summaries = self._format_ticker_summary(ticker_data)

        prompt = DAILY_BRIEFING_PROMPT.format(
            market_context=market_context,
            ticker_summaries=ticker_summaries,
        )

        response = self.llm.invoke(prompt)

        # Handle both string and message responses
        if hasattr(response, "content"):
            content = response.content
            # Gemini with thinking returns list of dicts
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ) or str(content)
            return str(content)
        return str(response)
