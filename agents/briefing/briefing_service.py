"""Daily briefing service — aggregates technical + news data and synthesizes via LLM.

Uses PydanticOutputParser for structured, deterministic output (same pattern
as SECDocumentProcessor in sec_llm_models.py).
"""

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.llm_utils import parse_llm_response
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
    news_url: Optional[str] = Field(
        default=None,
        description="URL to the most relevant news article (injected post-LLM, not generated)"
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
            if t.news_url:
                lines.append(f"- **News:** [{t.news_summary}]({t.news_url})")
            else:
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


@dataclass(frozen=True)
class BriefingResult:
    """Container for a briefing analysis plus the LLM's chain-of-thought."""
    analysis: DailyBriefingAnalysis
    thinking: str


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

    def generate(self, tickers: list[str]) -> BriefingResult:
        """Build briefing for the given tickers.

        Returns a BriefingResult containing the structured analysis and
        the LLM's chain-of-thought reasoning.

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

        result = self._synthesize(ticker_data, regime_data, news_data)

        _briefing_cache = (cache_key, now, result)
        return result

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
                self._merge_indicators(entry, indicators)

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

    def _merge_indicators(self, entry: dict[str, Any], indicators: dict[str, Any]) -> None:
        """Copy RSI, MACD, and ADX values from indicators dict into the entry dict."""
        if "rsi" in indicators:
            entry["rsi"] = round(indicators["rsi"].get("current", 0), 1)
            entry["rsi_signal"] = indicators["rsi"].get("signal", "neutral")

        if "macd" in indicators:
            entry["macd_signal"] = indicators["macd"].get("signal", "neutral")
            entry["macd_histogram"] = round(indicators["macd"].get("histogram", 0), 4)

        if "adx" in indicators:
            entry["adx"] = indicators["adx"].get("adx", 0)
            entry["trend_strength"] = indicators["adx"].get("trend_strength", "unknown")

    def _get_market_regime(self) -> dict[str, Any]:
        """Get current market regime from SPY/VIX."""
        try:
            from agents.market_analysis.regime_detector import MarketRegimeDetector

            detector = MarketRegimeDetector()
            return detector.detect_regime()
        except Exception:
            return {"error": "Could not determine market regime"}

    def _gather_news(self, tickers: list[str]) -> dict[str, dict[str, str]]:
        """Fetch recent news per ticker via Tavily.

        Returns a dict mapping ticker -> {"summary": str, "url": str | None}.
        The summary is passed to the LLM prompt; the URL is injected into
        the Pydantic model after the LLM call (so URLs are never hallucinated).
        """
        empty = {"summary": "No news available (Tavily not configured)", "url": None}
        if not self.tavily_api_key:
            return {t: empty for t in tickers}

        from langchain_tavily import TavilySearch

        news: dict[str, dict[str, str]] = {}
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
                    news[ticker] = {"summary": result, "url": None}
                elif isinstance(result, dict):
                    answer = result.get("answer", "")
                    results_list = result.get("results", [])

                    # Extract top URL from the first result
                    top_url = results_list[0].get("url") if results_list else None

                    headlines = [r.get("title", "") for r in results_list[:3]]
                    parts = []
                    if answer:
                        parts.append(answer)
                    if headlines:
                        parts.append("Headlines: " + "; ".join(headlines))

                    summary = " | ".join(parts) if parts else "No recent news"
                    news[ticker] = {"summary": summary, "url": top_url}
                else:
                    news[ticker] = {"summary": str(result), "url": None}
            except Exception:
                news[ticker] = {"summary": "News retrieval failed", "url": None}

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

    def _format_news_context(self, news: dict[str, dict[str, str]]) -> str:
        """Format news summaries for the prompt (URLs excluded — injected post-LLM)."""
        lines = []
        for ticker, data in news.items():
            lines.append(f"{ticker}: {data['summary']}")
        return "\n".join(lines)

    # --- Synthesis --------------------------------------------------------

    def _synthesize(
        self,
        ticker_data: list[dict[str, Any]],
        regime_data: dict[str, Any],
        news_data: dict[str, dict[str, str]],
    ) -> BriefingResult:
        """Invoke LLM, capture thinking, then parse structured output.

        The chain is split into two steps so we can extract the LLM's
        chain-of-thought reasoning before the parser consumes the response:
          1. prompt | llm  →  raw AIMessage (may contain thinking blocks)
          2. parser.invoke(text)  →  DailyBriefingAnalysis

        URLs are injected into the parsed model after the LLM call so the
        model never has the opportunity to hallucinate them.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", DAILY_BRIEFING_SYSTEM_PROMPT),
            ("user", DAILY_BRIEFING_USER_TEMPLATE),
        ])

        # Step 1: prompt | llm — get raw response with thinking
        prompt_chain = prompt.partial(
            market_context=self._format_market_context(regime_data),
            ticker_summaries=self._format_ticker_summary(ticker_data),
            news_context=self._format_news_context(news_data),
            format_instructions=self.parser.get_format_instructions(),
        )
        raw_response = (prompt_chain | self.llm).invoke({})

        # Step 2: separate thinking from text using the reusable module
        parsed = parse_llm_response(raw_response)

        # Step 3: parse the text content into the Pydantic model
        analysis = self.parser.invoke(parsed.text)

        # Inject real URLs from Tavily (post-LLM, never hallucinated)
        url_map = {t: data.get("url") for t, data in news_data.items()}
        for ticker_briefing in analysis.tickers:
            ticker_briefing.news_url = url_map.get(ticker_briefing.ticker)

        return BriefingResult(analysis=analysis, thinking=parsed.thinking)
