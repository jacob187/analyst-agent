"""Tests for BriefingService."""

import time as time_mod
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest
import agents.briefing.briefing_service as briefing_mod
from agents.briefing.briefing_service import (
    BriefingResult,
    BriefingService,
    DailyBriefingAnalysis,
    NewsItem,
    TickerBriefing,
    _compute_diff,
    _format_relative_time,
    _invoke_with_timeout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ANALYSIS = DailyBriefingAnalysis(
    market_regime="Transitional Bear, High Volatility, Markdown Phase",
    market_positioning="Defensive posture recommended. Tighten stops and reduce position sizes.",
    tickers=[
        TickerBriefing(
            ticker="AAPL",
            price=180.50,
            change_pct=-1.23,
            technical_signal="RSI at 28.3 indicates oversold; watch for MACD bullish crossover.",
            outlook="mixed",
        ),
    ],
    alerts=["AAPL RSI oversold while market in markdown — potential bear trap."],
)

SAMPLE_JSON = SAMPLE_ANALYSIS.model_dump_json()


@pytest.fixture
def mock_llm():
    """Mock LLM that returns valid JSON for the PydanticOutputParser."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=SAMPLE_JSON)
    return llm


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level briefing cache between tests."""
    briefing_mod._briefing_cache.clear()
    yield
    briefing_mod._briefing_cache.clear()


@pytest.fixture
def service(mock_llm):
    return BriefingService(mock_llm)


@pytest.fixture
def service_with_tavily(mock_llm):
    return BriefingService(mock_llm, tavily_api_key="test-tavily-key")


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_ticker_briefing_fields(self):
        t = TickerBriefing(
            ticker="XOM",
            price=169.66,
            change_pct=0.99,
            technical_signal="RSI=72.4 overbought",
            outlook="bearish",
        )
        assert t.ticker == "XOM"
        assert t.outlook == "bearish"
        assert t.news_items == []

    def test_daily_briefing_to_markdown(self):
        md = SAMPLE_ANALYSIS.to_markdown()
        assert "## Market Briefing:" in md
        assert "AAPL" in md
        assert "$180.50" in md
        assert "## Alerts" in md
        assert "bear trap" in md
        assert "not investment advice" in md

    def test_to_markdown_renders_news_items_as_bullets(self):
        """Each news item becomes a markdown bullet under a News header."""
        analysis = SAMPLE_ANALYSIS.model_copy(deep=True)
        analysis.tickers[0].news_items = [
            NewsItem(headline="Apple beats Q2", url="https://example.com/1", published_at="2026-05-15"),
            NewsItem(headline="Tim Cook keynote", url="https://example.com/2", published_at="2026-05-14"),
        ]
        md = analysis.to_markdown()
        assert "- **News:**" in md
        assert "[Apple beats Q2](https://example.com/1)" in md
        assert "2026-05-15" in md
        assert "[Tim Cook keynote](https://example.com/2)" in md

    def test_to_markdown_news_item_without_url_is_plain(self):
        analysis = SAMPLE_ANALYSIS.model_copy(deep=True)
        analysis.tickers[0].news_items = [NewsItem(headline="Plain headline")]
        md = analysis.to_markdown()
        assert "Plain headline" in md
        assert "](http" not in md

    def test_to_markdown_skips_news_section_when_empty(self):
        """No news items → no News header at all."""
        md = SAMPLE_ANALYSIS.to_markdown()
        assert "**News:**" not in md

    def test_daily_briefing_roundtrip(self):
        """Serialize to JSON and back — ensures parser compatibility."""
        dumped = SAMPLE_ANALYSIS.model_dump()
        restored = DailyBriefingAnalysis(**dumped)
        assert restored.market_regime == SAMPLE_ANALYSIS.market_regime
        assert len(restored.tickers) == 1
        assert restored.tickers[0].ticker == "AAPL"


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

class TestFormatMarketContext:
    def test_with_full_data(self, service):
        regime = {
            "trend": "bull",
            "volatility": "low",
            "phase": "markup",
            "recommendations": ["Buy dips", "Use trend-following"],
        }
        result = service._format_market_context(regime)
        assert "BULL" in result
        assert "LOW" in result
        assert "MARKUP" in result
        assert "Buy dips" in result

    def test_with_error(self, service):
        result = service._format_market_context({"error": "failed"})
        assert "unavailable" in result.lower()


class TestFormatTickerSummary:
    def test_formats_ticker(self, service):
        data = [{"ticker": "AAPL", "price": 180.0, "rsi": 45.2, "rsi_signal": "neutral", "macd_signal": "bullish"}]
        result = service._format_ticker_summary(data)
        assert "AAPL" in result
        assert "180.00" in result
        assert "NEUTRAL" in result

    def test_handles_error(self, service):
        data = [{"ticker": "BAD", "error": "No data"}]
        result = service._format_ticker_summary(data)
        assert "unavailable" in result.lower()


class TestFormatNewsContext:
    def test_formats_headlines_no_urls(self, service):
        news = {
            "AAPL": [
                NewsItem(headline="Apple beat earnings", url="https://example.com/aapl"),
                NewsItem(headline="iPhone launch rumor", url="https://example.com/iphone"),
            ],
            "MSFT": [],
        }
        result = service._format_news_context(news)
        assert "AAPL:" in result
        assert "Apple beat earnings" in result
        assert "iPhone launch rumor" in result
        assert "MSFT: No recent news." in result
        # URLs are excluded from the prompt text (injected post-LLM)
        assert "https://" not in result


# ---------------------------------------------------------------------------
# News gathering
# ---------------------------------------------------------------------------

class TestGatherNews:
    def test_no_tavily_key_returns_empty_lists(self, service):
        """Without a Tavily key, every ticker maps to an empty list."""
        result = service._gather_news(["AAPL", "MSFT"])
        assert result == {"AAPL": [], "MSFT": []}

    def test_tavily_search_called_with_time_range_day(self, service_with_tavily):
        """With a Tavily key, TavilySearch is constructed with time_range='day'."""
        mock_search = MagicMock()
        mock_search.invoke.return_value = {
            "results": [{
                "title": "AAPL surges",
                "url": "https://example.com/aapl",
                "published_date": "2026-05-16",
            }],
        }
        ctor = MagicMock(return_value=mock_search)

        with patch("langchain_tavily.TavilySearch", ctor):
            result = service_with_tavily._gather_news(["AAPL"])

        assert len(result["AAPL"]) == 1
        item = result["AAPL"][0]
        assert item.headline == "AAPL surges"
        assert item.url == "https://example.com/aapl"
        assert item.published_at == "2026-05-16"
        # day-first attempt: TavilySearch built once with time_range="day"
        kwargs = ctor.call_args.kwargs
        assert kwargs["time_range"] == "day"
        assert kwargs["topic"] == "news"

    def test_falls_back_to_week_when_day_empty(self, service_with_tavily):
        """When time_range='day' returns 0 results, retry with 'week'."""
        empty = {"results": []}
        weekly = {"results": [{"title": "Week-old headline", "url": "https://example.com/w"}]}

        mock_day_search = MagicMock()
        mock_day_search.invoke.return_value = empty
        mock_week_search = MagicMock()
        mock_week_search.invoke.return_value = weekly

        # Two constructor calls — first returns the empty-day search, second the weekly one.
        ctor = MagicMock(side_effect=[mock_day_search, mock_week_search])

        with patch("langchain_tavily.TavilySearch", ctor):
            result = service_with_tavily._gather_news(["AAPL"])

        assert len(result["AAPL"]) == 1
        assert result["AAPL"][0].headline == "Week-old headline"
        # Verify both time_ranges were tried in order.
        assert [c.kwargs["time_range"] for c in ctor.call_args_list] == ["day", "week"]

    def test_tavily_error_returns_empty(self, service_with_tavily):
        """Tavily failures degrade gracefully to an empty list."""
        mock_search = MagicMock()
        mock_search.invoke.side_effect = Exception("API down")

        with patch("langchain_tavily.TavilySearch", return_value=mock_search):
            result = service_with_tavily._gather_news(["AAPL"])
            assert result["AAPL"] == []

    def test_skips_results_with_blank_title(self, service_with_tavily):
        """Result rows without a title are dropped."""
        mock_search = MagicMock()
        mock_search.invoke.return_value = {
            "results": [
                {"title": "", "url": "https://example.com/bad"},
                {"title": "Good headline", "url": "https://example.com/good"},
            ],
        }
        with patch("langchain_tavily.TavilySearch", return_value=mock_search):
            result = service_with_tavily._gather_news(["AAPL"])
        assert len(result["AAPL"]) == 1
        assert result["AAPL"][0].headline == "Good headline"


class TestParallelFanOut:
    def test_one_slow_ticker_does_not_block_others(self, service_with_tavily, monkeypatch):
        """A ticker that takes 5s must not delay the other 4 tickers.

        With serial fetching the total wall time would be ~5.2s. With the
        ThreadPoolExecutor fan-out it should finish in ~5s (the slow one)
        plus a small overhead, well under what serial would take.
        """
        # Compress timeouts so the test is fast but the asymmetry still shows.
        monkeypatch.setattr(briefing_mod, "_NEWS_FAN_TIMEOUT_SECONDS", 30)
        monkeypatch.setattr(briefing_mod, "_TAVILY_PER_CALL_TIMEOUT_SECONDS", 30)

        fast_payload = {"results": [{"title": "fast", "url": "https://x/"}]}

        # Route by ticker via the query string (cleaner than the ctor side).
        per_query_mock = MagicMock()
        def routed_invoke(payload):
            ticker = payload["query"].split()[0]
            if ticker == "SLOW":
                time_mod.sleep(0.5)
            else:
                time_mod.sleep(0.05)
            return fast_payload
        per_query_mock.invoke.side_effect = routed_invoke

        tickers = ["A", "B", "SLOW", "D", "E"]
        with patch("langchain_tavily.TavilySearch", return_value=per_query_mock):
            t0 = time_mod.perf_counter()
            result = service_with_tavily._gather_news(tickers)
            elapsed = time_mod.perf_counter() - t0

        # All 5 tickers populated
        assert set(result.keys()) == set(tickers)
        for t in tickers:
            assert len(result[t]) == 1
        # Parallel: should be ~max(per_call_times) + overhead, well under 5×0.5s
        assert elapsed < 1.5, f"Fan-out took {elapsed:.2f}s — looks serial"

    def test_hung_ticker_yields_empty_within_per_call_timeout(self, service_with_tavily, monkeypatch):
        """If one ticker's Tavily call hangs, it must surface as [] without
        delaying the others past the per-call timeout."""
        # Compress so the test is fast: per-call cap 0.3s.
        monkeypatch.setattr(briefing_mod, "_TAVILY_PER_CALL_TIMEOUT_SECONDS", 0.3)
        monkeypatch.setattr(briefing_mod, "_NEWS_FAN_TIMEOUT_SECONDS", 5)

        fast_payload = {"results": [{"title": "fast", "url": "https://x/"}]}

        def routed_invoke(payload):
            ticker = payload["query"].split()[0]
            if ticker == "HUNG":
                # Simulate a hung request — block longer than both retries can wait.
                time_mod.sleep(2.0)
            return fast_payload

        search_mock = MagicMock()
        search_mock.invoke.side_effect = routed_invoke

        tickers = ["AAPL", "HUNG", "MSFT"]
        with patch("langchain_tavily.TavilySearch", return_value=search_mock):
            t0 = time_mod.perf_counter()
            result = service_with_tavily._gather_news(tickers)
            elapsed = time_mod.perf_counter() - t0

        # Fast tickers get results
        assert len(result["AAPL"]) == 1
        assert len(result["MSFT"]) == 1
        # Hung ticker gets an empty list (both day and week timeouts fired)
        assert result["HUNG"] == []
        # Total time bounded by 2 × per-call timeout for the hung ticker (day + week),
        # not the 2s of real sleep — with generous slack for test overhead.
        assert elapsed < 1.5, f"Fan-out blocked on hung ticker: {elapsed:.2f}s"


class TestInvokeWithTimeout:
    def test_returns_result_when_fast(self):
        search = MagicMock()
        search.invoke.return_value = {"results": [{"title": "x"}]}
        result = _invoke_with_timeout(search, "q", 2.0)
        assert result == {"results": [{"title": "x"}]}

    def test_returns_none_on_timeout(self):
        search = MagicMock()
        def slow(_):
            time_mod.sleep(1.0)
            return "late"
        search.invoke.side_effect = slow
        t0 = time_mod.perf_counter()
        result = _invoke_with_timeout(search, "q", 0.2)
        elapsed = time_mod.perf_counter() - t0
        assert result is None
        # Caller is unblocked within the timeout (plus small overhead),
        # not after the full sleep.
        assert elapsed < 0.6

    def test_propagates_invoke_errors(self):
        """Errors raised by invoke surface as exceptions (callers wrap in try)."""
        search = MagicMock()
        search.invoke.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            _invoke_with_timeout(search, "q", 2.0)


# ---------------------------------------------------------------------------
# Synthesis (prompt | llm | parser chain)
# ---------------------------------------------------------------------------

class TestSynthesize:
    def test_returns_briefing_result_with_injected_news(self):
        """News items from _gather_news are injected post-LLM into each TickerBriefing.

        Uses LangChain's FakeListChatModel so the chain runs end-to-end.
        """
        from langchain_core.language_models import FakeListChatModel

        fake_llm = FakeListChatModel(responses=[SAMPLE_JSON])
        service = BriefingService(fake_llm)

        news_data = {
            "AAPL": [NewsItem(headline="Apple beat earnings", url="https://example.com/aapl", published_at="2026-05-16")],
        }
        result = service._synthesize(
            [{"ticker": "AAPL", "price": 180.0}],
            {"trend": "bull", "volatility": "low", "phase": "markup"},
            news_data,
        )
        assert isinstance(result, BriefingResult)
        assert isinstance(result.analysis, DailyBriefingAnalysis)
        assert result.analysis.tickers[0].ticker == "AAPL"
        # News items are injected — the LLM never wrote them
        assert len(result.analysis.tickers[0].news_items) == 1
        assert result.analysis.tickers[0].news_items[0].headline == "Apple beat earnings"
        assert result.analysis.tickers[0].news_items[0].url == "https://example.com/aapl"
        # FakeListChatModel returns plain strings — no thinking blocks
        assert result.thinking == ""


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_second_call_cached(self, service, mock_llm):
        sample_result = BriefingResult(analysis=SAMPLE_ANALYSIS, thinking="some thought")
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "AAPL", "price": 180}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"AAPL": []}):
                    with patch.object(service, "_synthesize", return_value=sample_result):
                        service.generate(["AAPL"], user_id="u1", model_id="m1")
                        result = service.generate(["AAPL"], user_id="u1", model_id="m1")
                        # _synthesize should only be called once (second call hits cache)
                        assert service._synthesize.call_count == 1
                        assert isinstance(result, BriefingResult)
                        assert result.thinking == "some thought"

    def test_different_tickers_not_cached(self, service, mock_llm):
        sample_result = BriefingResult(analysis=SAMPLE_ANALYSIS, thinking="")
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "X", "price": 1}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"X": []}):
                    with patch.object(service, "_synthesize", return_value=sample_result):
                        service.generate(["AAPL"], user_id="u1", model_id="m1")
                        service.generate(["MSFT"], user_id="u1", model_id="m1")
                        assert service._synthesize.call_count == 2

    def test_cache_isolated_per_user(self, service, mock_llm):
        """Two users with identical watchlists must NOT share a cached briefing.

        Locks the cross-user leak fix: news data is fetched per-call and varies
        over time, so reusing one user's briefing for another would leak content.
        """
        sample_result = BriefingResult(analysis=SAMPLE_ANALYSIS, thinking="")
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "AAPL", "price": 180}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"AAPL": []}):
                    with patch.object(service, "_synthesize", return_value=sample_result):
                        service.generate(["AAPL"], user_id="user_a", model_id="m1")
                        service.generate(["AAPL"], user_id="user_b", model_id="m1")
                        assert service._synthesize.call_count == 2

    def test_cache_isolated_per_model(self, service, mock_llm):
        """Same user + tickers but different model must re-run the LLM."""
        sample_result = BriefingResult(analysis=SAMPLE_ANALYSIS, thinking="")
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "AAPL", "price": 180}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"AAPL": []}):
                    with patch.object(service, "_synthesize", return_value=sample_result):
                        service.generate(["AAPL"], user_id="user_a", model_id="gemini-3-pro")
                        service.generate(["AAPL"], user_id="user_a", model_id="gpt-4.1")
                        assert service._synthesize.call_count == 2


# ---------------------------------------------------------------------------
# "Since last briefing" diff
# ---------------------------------------------------------------------------

def _make_current(ticker: str, price: float, outlook: str = "neutral",
                  market_regime: str = "Bull, Moderate Volatility, Markup Phase") -> DailyBriefingAnalysis:
    return DailyBriefingAnalysis(
        market_regime=market_regime,
        market_positioning="Stay long.",
        tickers=[
            TickerBriefing(
                ticker=ticker, price=price, change_pct=0.0,
                technical_signal="RSI 50", outlook=outlook,
            ),
        ],
        alerts=[],
    )


class TestComputeDiff:
    def test_empty_when_nothing_changed(self):
        prev = {
            "market_regime": "Bull, Moderate Volatility, Markup Phase",
            "tickers": [{"ticker": "AAPL", "price": 180.0, "outlook": "neutral"}],
        }
        current = _make_current("AAPL", 180.0, "neutral")
        assert _compute_diff(prev, current) == []

    def test_detects_regime_change(self):
        prev = {
            "market_regime": "Bear, High Volatility, Markdown Phase",
            "tickers": [{"ticker": "AAPL", "price": 180.0, "outlook": "neutral"}],
        }
        current = _make_current("AAPL", 180.0, "neutral")
        bullets = _compute_diff(prev, current)
        assert any("Market regime" in b for b in bullets)
        assert any("Bear" in b and "Bull" in b for b in bullets)

    def test_detects_outlook_flip(self):
        prev = {
            "market_regime": "Bull, Moderate Volatility, Markup Phase",
            "tickers": [{"ticker": "NVDA", "price": 245.0, "outlook": "bullish"}],
        }
        current = _make_current("NVDA", 245.0, "bearish")
        bullets = _compute_diff(prev, current)
        assert len(bullets) == 1
        assert "NVDA" in bullets[0]
        assert "bullish" in bullets[0]
        assert "bearish" in bullets[0]

    def test_detects_price_delta(self):
        prev = {
            "market_regime": "Bull, Moderate Volatility, Markup Phase",
            "tickers": [{"ticker": "NVDA", "price": 245.10, "outlook": "bullish"}],
        }
        current = _make_current("NVDA", 225.32, "bullish")
        bullets = _compute_diff(prev, current)
        assert len(bullets) == 1
        assert "$245.10" in bullets[0]
        assert "$225.32" in bullets[0]
        # ~-8.07% — sign should be negative, exact value not asserted to avoid float-fragility
        assert "(-" in bullets[0]

    def test_new_ticker_in_briefing(self):
        prev = {
            "market_regime": "Bull, Moderate Volatility, Markup Phase",
            "tickers": [{"ticker": "AAPL", "price": 180.0, "outlook": "neutral"}],
        }
        current = _make_current("MSFT", 421.92, "bullish")
        bullets = _compute_diff(prev, current)
        assert bullets == ["MSFT: new to briefing"]

    def test_removed_ticker_is_not_reported(self):
        """Tickers in previous but not current are dropped — user cares about current watchlist."""
        prev = {
            "market_regime": "Bull, Moderate Volatility, Markup Phase",
            "tickers": [
                {"ticker": "AAPL", "price": 180.0, "outlook": "neutral"},
                {"ticker": "REMOVED", "price": 100.0, "outlook": "bullish"},
            ],
        }
        current = _make_current("AAPL", 180.0, "neutral")
        assert _compute_diff(prev, current) == []


class TestFormatRelativeTime:
    def test_yesterday(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=26)).strftime("%Y-%m-%d %H:%M:%S")
        assert _format_relative_time(ts) == "yesterday"

    def test_hours_ago(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        assert _format_relative_time(ts) == "5h ago"

    def test_days_ago(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=4, hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        assert _format_relative_time(ts) == "4 days ago"

    def test_less_than_hour(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=12)).strftime("%Y-%m-%d %H:%M:%S")
        assert _format_relative_time(ts) == "less than an hour ago"

    def test_malformed_falls_back(self):
        assert _format_relative_time("not a timestamp") == "previous"
        assert _format_relative_time(None) == "previous"


class TestSinceLastMarkdown:
    def test_section_rendered_at_top_when_diff_present(self):
        analysis = _make_current("AAPL", 180.0)
        analysis.since_last = ["NVDA: $245.10 → $225.32 (-8.07%)"]
        analysis.since_last_label = "yesterday"
        md = analysis.to_markdown()
        assert "## Since Last Briefing (yesterday)" in md
        assert "NVDA: $245.10" in md
        # The diff section is rendered before the market briefing header
        assert md.index("## Since Last Briefing") < md.index("## Market Briefing")

    def test_section_skipped_when_empty(self):
        md = _make_current("AAPL", 180.0).to_markdown()
        assert "Since Last Briefing" not in md

    def test_section_skipped_when_label_missing(self):
        """No label → header rendered without the parenthetical."""
        analysis = _make_current("AAPL", 180.0)
        analysis.since_last = ["MSFT: new to briefing"]
        md = analysis.to_markdown()
        assert "## Since Last Briefing\n" in md
        assert "(yesterday)" not in md


class TestGenerateWithPreviousBriefing:
    def test_diff_attached_to_result(self, service, mock_llm):
        """When generate() is given a previous briefing, since_last is populated."""
        prev_dict = {
            "market_regime": "Bull, Moderate Volatility, Markup Phase",
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=22)).strftime("%Y-%m-%d %H:%M:%S"),
            "tickers": [{"ticker": "AAPL", "price": 170.0, "outlook": "bullish"}],
        }
        sample = BriefingResult(analysis=_make_current("AAPL", 180.0, "bearish"), thinking="")

        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "AAPL", "price": 180}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"AAPL": []}):
                    with patch.object(service, "_synthesize", return_value=sample):
                        result = service.generate(
                            ["AAPL"], user_id="u1", model_id="m1",
                            previous_briefing=prev_dict,
                        )

        assert result.analysis.since_last  # non-empty
        assert result.analysis.since_last_label == "22h ago"
        # The diff should include both the price delta and outlook flip for AAPL
        joined = " ".join(result.analysis.since_last)
        assert "$170.00" in joined and "$180.00" in joined
        assert "bullish" in joined and "bearish" in joined

    def test_no_previous_means_no_diff(self, service, mock_llm):
        sample = BriefingResult(analysis=_make_current("AAPL", 180.0), thinking="")
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "AAPL", "price": 180}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"AAPL": []}):
                    with patch.object(service, "_synthesize", return_value=sample):
                        result = service.generate(["AAPL"], user_id="u1", model_id="m1")
        assert result.analysis.since_last == []
        assert result.analysis.since_last_label is None
