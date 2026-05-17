"""Daily briefing service — aggregates technical + news data and synthesizes via LLM.

Uses PydanticOutputParser for structured, deterministic output (same pattern
as SECDocumentProcessor in sec_llm_models.py).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from cachetools import TTLCache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.llm_utils import parse_llm_response
from agents.prompts import DAILY_BRIEFING_SYSTEM_PROMPT, DAILY_BRIEFING_USER_TEMPLATE

# Maximum number of news items shown per ticker. Tavily returns up to
# `max_results` per query; this trims the rendered list.
_MAX_NEWS_ITEMS_PER_TICKER = 3

# Per-call wall-clock cap on a single TavilySearch.invoke(). The underlying
# langchain-tavily wrapper calls requests.post() with no timeout, so we have
# to enforce it externally via a Future. Hung threads continue running until
# the OS-level socket timeout (~75s) but the caller is unblocked promptly.
_TAVILY_PER_CALL_TIMEOUT_SECONDS = 10

# Cap on the whole per-ticker fan-out across all parallel Tavily calls.
# If all tickers are healthy, this is irrelevant; if one ticker hangs, the
# other 9 still deliver fresh news and the briefing returns quickly.
_NEWS_FAN_TIMEOUT_SECONDS = 30

# Parallel workers for the news fan-out. Bounded to keep Tavily rate-limit
# friendly while still saturating most watchlists in one round-trip.
_TAVILY_MAX_WORKERS = 5


# ---------------------------------------------------------------------------
# Pydantic models — define the exact shape of the LLM output
# ---------------------------------------------------------------------------

class NewsItem(BaseModel):
    """A single news article surfaced by Tavily for a ticker.

    Injected post-LLM (never generated) so headlines and URLs are provably real.
    """

    headline: str = Field(description="Article title from the news source")
    url: Optional[str] = Field(default=None, description="Direct link to the article")
    published_at: Optional[str] = Field(
        default=None,
        description="Publication date if Tavily returned one (often ISO format)",
    )


class TickerBriefing(BaseModel):
    """Structured analysis for a single ticker in the daily briefing."""

    ticker: str = Field(description="Ticker symbol")
    price: float = Field(description="Current price in USD")
    change_pct: float = Field(description="Daily change percentage")
    technical_signal: str = Field(
        description="One-sentence summary of the most actionable technical signal "
        "(RSI, MACD crossover, pattern, or ADX trend strength)"
    )
    news_items: list[NewsItem] = Field(
        default_factory=list,
        description=(
            "DO NOT GENERATE THIS FIELD. Leave it as an empty list. "
            "Real news items from Tavily are injected after the LLM call."
        ),
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
    since_last: list[str] = Field(
        default_factory=list,
        description=(
            "DO NOT GENERATE THIS FIELD. Leave it empty. "
            "Diff bullets vs the user's previous briefing are computed and "
            "injected after the LLM call."
        ),
    )
    since_last_label: Optional[str] = Field(
        default=None,
        description="Relative time label for the previous briefing (e.g. 'yesterday', '3h ago')",
    )

    def to_markdown(self) -> str:
        """Render the structured briefing as markdown for the frontend."""
        lines = []

        if self.since_last:
            header = "## Since Last Briefing"
            if self.since_last_label:
                header += f" ({self.since_last_label})"
            lines.append(header)
            lines.append("")
            for bullet in self.since_last:
                lines.append(f"- {bullet}")
            lines.append("")

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
            if t.news_items:
                lines.append("- **News:**")
                for item in t.news_items[:_MAX_NEWS_ITEMS_PER_TICKER]:
                    suffix = f" — {item.published_at}" if item.published_at else ""
                    if item.url:
                        lines.append(f"  - [{item.headline}]({item.url}){suffix}")
                    else:
                        lines.append(f"  - {item.headline}{suffix}")
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

# Keyed on (user_id, model_id, sorted_tickers). News data is fetched per-call
# and varies over time, so two users with the same watchlist must NOT share
# a cached briefing — that was the cross-user leak this cache replaces.
_briefing_cache: TTLCache = TTLCache(maxsize=64, ttl=900)


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

    def generate(
        self,
        tickers: list[str],
        user_id: str,
        model_id: str,
        previous_briefing: Optional[dict] = None,
    ) -> BriefingResult:
        """Build briefing for the given tickers.

        Returns a BriefingResult containing the structured analysis and
        the LLM's chain-of-thought reasoning.

        Cache key is (user_id, model_id, sorted_tickers); two users with
        identical watchlists never share a generated briefing.

        If ``previous_briefing`` is provided (the dict shape returned by
        ``api.db.get_recent_briefings``), a "since last briefing" diff is
        attached to the resulting analysis. Pass ``None`` if no prior
        briefing exists.
        """
        cache_key = (user_id, model_id, tuple(sorted(tickers)))
        cached = _briefing_cache.get(cache_key)
        if cached is not None:
            return cached

        ticker_data = self._gather_ticker_data(tickers)
        regime_data = self._get_market_regime()
        news_data = self._gather_news(tickers)

        result = self._synthesize(ticker_data, regime_data, news_data)

        if previous_briefing:
            diff_lines = _compute_diff(previous_briefing, result.analysis)
            if diff_lines:
                result.analysis.since_last = diff_lines
                result.analysis.since_last_label = _format_relative_time(
                    previous_briefing.get("created_at")
                )

        _briefing_cache[cache_key] = result
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

    def _gather_news(self, tickers: list[str]) -> dict[str, list[NewsItem]]:
        """Fetch recent news per ticker via Tavily, in parallel.

        Returns a dict mapping ticker -> list of NewsItem (up to
        _MAX_NEWS_ITEMS_PER_TICKER), preserving the input ticker order.
        News is injected into the parsed Pydantic model after the LLM call
        so the model never has the opportunity to hallucinate headlines.

        Freshness: each ticker first attempts time_range="day"; if that
        returns zero results, retries with time_range="week" so the briefing
        isn't empty on quiet days.

        Hung Tavily calls are bounded two ways:
          - Per-call wall-clock timeout (_TAVILY_PER_CALL_TIMEOUT_SECONDS)
          - Whole-fan timeout (_NEWS_FAN_TIMEOUT_SECONDS); tickers not done
            by then surface as empty lists, the briefing still ships.
        """
        if not self.tavily_api_key or not tickers:
            return {t: [] for t in tickers}

        from concurrent.futures import (
            ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed,
        )
        from langchain_tavily import TavilySearch

        news: dict[str, list[NewsItem]] = {t: [] for t in tickers}
        pool = ThreadPoolExecutor(max_workers=min(len(tickers), _TAVILY_MAX_WORKERS))
        try:
            futures = {
                pool.submit(self._fetch_ticker_news, TavilySearch, t): t
                for t in tickers
            }
            try:
                for fut in as_completed(futures, timeout=_NEWS_FAN_TIMEOUT_SECONDS):
                    ticker = futures[fut]
                    try:
                        news[ticker] = fut.result()
                    except Exception:
                        news[ticker] = []
            except FuturesTimeout:
                # Stragglers stay as [] in the pre-populated dict.
                pass
        finally:
            # cancel_futures=True (3.9+) cancels not-yet-started work; in-flight
            # requests stay running until they return/error at the OS socket level.
            pool.shutdown(wait=False, cancel_futures=True)
        return news

    def _fetch_ticker_news(self, tavily_cls: Any, ticker: str) -> list[NewsItem]:
        """Run Tavily for one ticker with a day→week freshness fallback."""
        query = f"{ticker} stock latest news"
        for time_range in ("day", "week"):
            try:
                search = tavily_cls(
                    tavily_api_key=self.tavily_api_key,
                    max_results=_MAX_NEWS_ITEMS_PER_TICKER,
                    search_depth="basic",
                    include_answer=False,
                    topic="news",
                    time_range=time_range,
                )
                result = _invoke_with_timeout(
                    search, query, _TAVILY_PER_CALL_TIMEOUT_SECONDS,
                )
            except Exception:
                return []
            if result is None:
                # Per-call timeout fired; treat as a soft failure and try
                # the next time_range (which may also time out → return []).
                continue

            items = self._parse_tavily_results(result)
            if items:
                return items
        return []

    @staticmethod
    def _parse_tavily_results(result: Any) -> list[NewsItem]:
        """Turn a TavilySearch response into a list of NewsItem."""
        if not isinstance(result, dict):
            return []
        results_list = result.get("results", []) or []
        items: list[NewsItem] = []
        for r in results_list[:_MAX_NEWS_ITEMS_PER_TICKER]:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            if not title:
                continue
            items.append(NewsItem(
                headline=title,
                url=r.get("url") or None,
                published_at=r.get("published_date") or None,
            ))
        return items

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

    def _format_news_context(self, news: dict[str, list[NewsItem]]) -> str:
        """Format news headlines for the prompt (URLs excluded — injected post-LLM).

        The LLM sees compact bullet lists per ticker so it can reason about
        sentiment for the outlook field, but it never writes the headlines.
        """
        lines = []
        for ticker, items in news.items():
            if not items:
                lines.append(f"{ticker}: No recent news.")
                continue
            lines.append(f"{ticker}:")
            for item in items[:_MAX_NEWS_ITEMS_PER_TICKER]:
                lines.append(f"  - {item.headline}")
        return "\n".join(lines)

    # --- Synthesis --------------------------------------------------------

    def _synthesize(
        self,
        ticker_data: list[dict[str, Any]],
        regime_data: dict[str, Any],
        news_data: dict[str, list[NewsItem]],
    ) -> BriefingResult:
        """Invoke LLM, capture thinking, then parse structured output.

        The chain is split into two steps so we can extract the LLM's
        chain-of-thought reasoning before the parser consumes the response:
          1. prompt | llm  →  raw AIMessage (may contain thinking blocks)
          2. parser.invoke(text)  →  DailyBriefingAnalysis

        News items are injected into the parsed model after the LLM call so
        the model never has the opportunity to hallucinate them.
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

        # Inject real news items from Tavily (post-LLM, never hallucinated).
        # The LLM may have emitted news_items=[]; overwrite either way.
        for ticker_briefing in analysis.tickers:
            ticker_briefing.news_items = news_data.get(ticker_briefing.ticker, [])

        return BriefingResult(analysis=analysis, thinking=parsed.thinking)


# ---------------------------------------------------------------------------
# TavilySearch wall-clock timeout helper
# ---------------------------------------------------------------------------

def _invoke_with_timeout(search: Any, query: str, timeout_s: float) -> Optional[Any]:
    """Call ``search.invoke({"query": query})`` with an external timeout.

    Returns the result, or ``None`` if the call did not complete within
    ``timeout_s`` seconds. Necessary because langchain-tavily's underlying
    ``requests.post()`` has no timeout argument — without this wrapper a hung
    Tavily endpoint would block the calling worker for the full OS-level
    socket timeout (~75s) per attempt.

    The hung thread is not killed (Python cannot cancel an in-flight sync
    request), but it is detached from the caller via ``shutdown(wait=False,
    cancel_futures=True)`` so the briefing can ship on time.
    """
    from concurrent.futures import (
        ThreadPoolExecutor, TimeoutError as FuturesTimeout,
    )

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(search.invoke, {"query": query})
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# "Since last briefing" diff
# ---------------------------------------------------------------------------

def _compute_diff(previous: dict, current: DailyBriefingAnalysis) -> list[str]:
    """Build a list of human-readable bullets describing what changed.

    ``previous`` is a dict in the shape returned by ``get_recent_briefings``.
    Tickers present only in the previous briefing are ignored (the user
    cares about their current watchlist, not removed positions).
    Returns an empty list if nothing changed.
    """
    bullets: list[str] = []

    prev_regime = (previous.get("market_regime") or "").strip()
    if prev_regime and prev_regime != current.market_regime:
        bullets.append(f"Market regime: {prev_regime} → {current.market_regime}")

    prev_tickers = {t["ticker"]: t for t in previous.get("tickers", []) if t.get("ticker")}

    for t in current.tickers:
        prev = prev_tickers.get(t.ticker)
        if prev is None:
            bullets.append(f"{t.ticker}: new to briefing")
            continue

        parts: list[str] = []

        prev_price = prev.get("price")
        if isinstance(prev_price, (int, float)) and prev_price > 0:
            delta_pct = (t.price - prev_price) / prev_price * 100
            # Skip trivial moves (<0.05%) — the line is just noise otherwise
            # and would clutter near-immediate re-runs after a cache miss.
            if abs(delta_pct) >= 0.05:
                parts.append(f"${prev_price:.2f} → ${t.price:.2f} ({delta_pct:+.2f}%)")

        prev_outlook = (prev.get("outlook") or "").lower()
        if prev_outlook and prev_outlook != t.outlook:
            parts.append(f"outlook {prev_outlook} → {t.outlook}")

        if parts:
            bullets.append(f"{t.ticker}: " + ", ".join(parts))

    return bullets


def _format_relative_time(created_at: Any) -> str:
    """Render a SQLite CURRENT_TIMESTAMP string as 'yesterday' / '3h ago' / '2 days ago'.

    Falls back to 'previous' on parse failure rather than crashing the
    briefing render path.
    """
    if not isinstance(created_at, str):
        return "previous"
    try:
        # SQLite's CURRENT_TIMESTAMP is UTC, 'YYYY-MM-DD HH:MM:SS'
        ts = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return "previous"

    delta = datetime.now(timezone.utc) - ts
    total_seconds = max(delta.total_seconds(), 0)
    if total_seconds < 3600:
        return "less than an hour ago"
    hours = total_seconds / 3600
    if hours < 24:
        return f"{int(hours)}h ago"
    days = int(hours / 24)
    if days == 1:
        return "yesterday"
    return f"{days} days ago"
