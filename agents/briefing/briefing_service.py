"""Daily briefing service — aggregates technical + news data and synthesizes via LLM.

Uses PydanticOutputParser for structured, deterministic output (same pattern
as SECDocumentProcessor in sec_llm_models.py).
"""

import time
from typing import Any, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts import DAILY_BRIEFING_SYSTEM_PROMPT, DAILY_BRIEFING_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Pydantic models — define the exact shape of the LLM output
# ---------------------------------------------------------------------------

class TickerBriefing(BaseModel):
    """Structured analysis for a single ticker in the daily briefing."""

    ticker: str = Field(description="Ticker symbol")
    price: float = Field(description="Current price in USD")
    change_pct: float = Field(description="Daily change percentage")
    technical_signal: str = Field(
        description="One-sentence summary of the most actionable technical signal "
        "(RSI, MACD crossover, pattern, or ADX trend strength)"
    )
    news_summary: str = Field(
        description="One-sentence summary of the most relevant recent news or "
        "'No recent news' if unavailable"
    )
    outlook: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        description="One of: bullish, bearish, neutral, or mixed"
    )


class DailyBriefingAnalysis(BaseModel):
    """Full structured daily briefing for a watchlist."""

    market_regime: str = Field(
        description="Current market regime summary (trend, volatility, phase)"
    )
    market_positioning: str = Field(
        description="What the current regime means for positioning (1-2 sentences)"
    )
    tickers: list[TickerBriefing] = Field(
        description="Per-ticker analysis for each watchlist ticker"
    )
    alerts: list[str] = Field(
        description="List of critical alerts: conflicts between indicators, "
        "extreme readings, or news events that demand immediate attention"
    )
    disclaimer: str = Field(
        default="This is analysis commentary, not investment advice.",
        description="Standard disclaimer"
    )

    def to_markdown(self) -> str:
        """Render the structured briefing as markdown for the frontend."""
        lines = []

        lines.append(f"## Market Briefing: {self.market_regime}")
        lines.append("")
        lines.append(self.market_positioning)
        lines.append("")

        lines.append("## Ticker Analysis")
        lines.append("")
        for t in self.tickers:
            direction = {"bullish": "+", "bearish": "-", "neutral": "~", "mixed": "?"}
            icon = direction.get(t.outlook, "~")
            lines.append(f"### {t.ticker} — ${t.price:.2f} ({t.change_pct:+.2f}%) [{icon} {t.outlook.upper()}]")
            lines.append(f"- **Technical:** {t.technical_signal}")
            lines.append(f"- **News:** {t.news_summary}")
            lines.append("")

        if self.alerts:
            lines.append("## Alerts")
            lines.append("")
            for alert in self.alerts:
                lines.append(f"- {alert}")
            lines.append("")

        lines.append(f"*{self.disclaimer}*")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_briefing_cache: tuple | None = None
_BRIEFING_CACHE_TTL = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class BriefingService:
    """Generate a structured daily market briefing for a watchlist.

    Follows the same prompt | llm | parser chain pattern used in
    SECDocumentProcessor (agents/sec_workflow/sec_llm_models.py).
    """

    def __init__(self, llm: BaseChatModel, tavily_api_key: Optional[str] = None):
        self.llm = llm
        self.tavily_api_key = tavily_api_key
        self.parser = PydanticOutputParser(pydantic_object=DailyBriefingAnalysis)

    def generate(self, tickers: list[str]) -> DailyBriefingAnalysis:
        """Build briefing for the given tickers. Returns structured Pydantic model.

        Uses a 15-minute in-memory cache keyed on the sorted ticker list.
        """
        global _briefing_cache
        now = time.monotonic()
        cache_key = tuple(sorted(tickers))

        if (
            _briefing_cache is not None
            and _briefing_cache[0] == cache_key
            and (now - _briefing_cache[1]) < _BRIEFING_CACHE_TTL
        ):
            return _briefing_cache[2]

        ticker_data = self._gather_ticker_data(tickers)
        regime_data = self._get_market_regime()
        news_data = self._gather_news(tickers)

        briefing = self._synthesize(ticker_data, regime_data, news_data)

        _briefing_cache = (cache_key, now, briefing)
        return briefing

    # --- Data gathering ---------------------------------------------------

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

                entry["price"] = round(float(hist["Close"].iloc[-1]), 2)
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    entry["change_pct"] = round(
                        (entry["price"] - prev) / prev * 100, 2
                    )

                ti = TechnicalIndicators(ticker)
                indicators = ti.calculate_all_indicators(hist)

                if "rsi" in indicators:
                    entry["rsi"] = round(indicators["rsi"].get("current", 0), 1)
                    entry["rsi_signal"] = indicators["rsi"].get("signal", "neutral")

                if "macd" in indicators:
                    entry["macd_signal"] = indicators["macd"].get("signal", "neutral")
                    entry["macd_histogram"] = round(
                        indicators["macd"].get("histogram", 0), 4
                    )

                if "adx" in indicators:
                    entry["adx"] = indicators["adx"].get("adx", 0)
                    entry["trend_strength"] = indicators["adx"].get(
                        "trend_strength", "unknown"
                    )

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

    def _gather_news(self, tickers: list[str]) -> dict[str, str]:
        """Fetch recent news headlines per ticker via Tavily.

        Returns a dict mapping ticker -> news summary string.
        Gracefully degrades when Tavily is unavailable.
        """
        if not self.tavily_api_key:
            return {t: "No news available (Tavily not configured)" for t in tickers}

        from langchain_tavily import TavilySearch

        news: dict[str, str] = {}
        search = TavilySearch(
            tavily_api_key=self.tavily_api_key,
            max_results=3,
            search_depth="basic",
            include_answer=True,
            topic="news",
        )

        for ticker in tickers:
            try:
                result = search.invoke({"query": f"{ticker} stock latest news"})

                if isinstance(result, str):
                    news[ticker] = result
                elif isinstance(result, dict):
                    answer = result.get("answer", "")
                    headlines = [
                        r.get("title", "")
                        for r in result.get("results", [])[:3]
                    ]
                    parts = []
                    if answer:
                        parts.append(answer)
                    if headlines:
                        parts.append("Headlines: " + "; ".join(headlines))
                    news[ticker] = " | ".join(parts) if parts else "No recent news"
                else:
                    news[ticker] = str(result)
            except Exception:
                news[ticker] = "News retrieval failed"

        return news

    # --- Formatting helpers -----------------------------------------------

    def _format_market_context(self, regime: dict[str, Any]) -> str:
        """Format market regime data for the prompt."""
        if "error" in regime:
            return "Market regime data unavailable."

        lines = []
        lines.append(f"Trend: {str(regime.get('trend', 'unknown')).upper()}")
        lines.append(f"Volatility: {str(regime.get('volatility', 'unknown')).upper()}")
        lines.append(f"Phase: {str(regime.get('phase', 'unknown')).upper()}")

        recs = regime.get("recommendations", [])
        if recs:
            lines.append("Strategy: " + "; ".join(recs[:3]))

        return "\n".join(lines)

    def _format_ticker_summary(self, data: list[dict[str, Any]]) -> str:
        """Format ticker technical data for the prompt."""
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

    def _format_news_context(self, news: dict[str, str]) -> str:
        """Format news data for the prompt."""
        lines = []
        for ticker, summary in news.items():
            lines.append(f"{ticker}: {summary}")
        return "\n".join(lines)

    # --- Synthesis --------------------------------------------------------

    def _synthesize(
        self,
        ticker_data: list[dict[str, Any]],
        regime_data: dict[str, Any],
        news_data: dict[str, str],
    ) -> DailyBriefingAnalysis:
        """Build prompt chain and invoke LLM with Pydantic output parser.

        Uses the same prompt | llm | parser pattern as SECDocumentProcessor.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", DAILY_BRIEFING_SYSTEM_PROMPT),
            ("user", DAILY_BRIEFING_USER_TEMPLATE),
        ])

        chain = (
            prompt.partial(
                market_context=self._format_market_context(regime_data),
                ticker_summaries=self._format_ticker_summary(ticker_data),
                news_context=self._format_news_context(news_data),
                format_instructions=self.parser.get_format_instructions(),
            )
            | self.llm
            | self.parser
        )

        return chain.invoke({})
