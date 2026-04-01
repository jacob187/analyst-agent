"""Tests for BriefingService."""

from unittest.mock import MagicMock, patch
import pytest
import agents.briefing.briefing_service as briefing_mod
from agents.briefing.briefing_service import (
    BriefingResult,
    BriefingService,
    DailyBriefingAnalysis,
    TickerBriefing,
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
            news_summary="Apple announced Q2 earnings beat expectations.",
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
    briefing_mod._briefing_cache = None
    yield
    briefing_mod._briefing_cache = None


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
            news_summary="No recent news",
            outlook="bearish",
        )
        assert t.ticker == "XOM"
        assert t.outlook == "bearish"

    def test_daily_briefing_to_markdown(self):
        md = SAMPLE_ANALYSIS.to_markdown()
        assert "## Market Briefing:" in md
        assert "AAPL" in md
        assert "$180.50" in md
        assert "## Alerts" in md
        assert "bear trap" in md
        assert "not investment advice" in md

    def test_to_markdown_with_news_url(self):
        """When news_url is set, the news line renders as a markdown link."""
        analysis = SAMPLE_ANALYSIS.model_copy(deep=True)
        analysis.tickers[0].news_url = "https://example.com/aapl-earnings"
        md = analysis.to_markdown()
        assert "[Apple announced Q2 earnings beat expectations.](https://example.com/aapl-earnings)" in md

    def test_to_markdown_without_news_url(self):
        """When news_url is None, the news line is plain text (no link)."""
        md = SAMPLE_ANALYSIS.to_markdown()
        assert "**News:** Apple announced" in md
        assert "](http" not in md

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
    def test_formats_news(self, service):
        news = {
            "AAPL": {"summary": "Apple beat earnings", "url": "https://example.com/aapl"},
            "MSFT": {"summary": "No recent news", "url": None},
        }
        result = service._format_news_context(news)
        assert "AAPL: Apple beat earnings" in result
        assert "MSFT: No recent news" in result
        # URLs are excluded from the prompt text (injected post-LLM)
        assert "https://" not in result


# ---------------------------------------------------------------------------
# News gathering
# ---------------------------------------------------------------------------

class TestGatherNews:
    def test_no_tavily_key_returns_placeholder(self, service):
        """Without a Tavily key, every ticker gets a placeholder message."""
        result = service._gather_news(["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "not configured" in result["AAPL"]["summary"].lower()
        assert result["AAPL"]["url"] is None

    def test_tavily_search_called(self, service_with_tavily):
        """With a Tavily key, TavilySearch is invoked for each ticker."""
        mock_search = MagicMock()
        mock_search.invoke.return_value = {
            "answer": "Apple stock rose 3%",
            "results": [{"title": "AAPL surges", "url": "https://example.com/aapl"}],
        }

        with patch("langchain_tavily.TavilySearch", return_value=mock_search):
            result = service_with_tavily._gather_news(["AAPL"])
            assert "Apple stock rose 3%" in result["AAPL"]["summary"]
            assert result["AAPL"]["url"] == "https://example.com/aapl"
            mock_search.invoke.assert_called_once()

    def test_tavily_error_handled(self, service_with_tavily):
        """Tavily failures degrade gracefully."""
        mock_search = MagicMock()
        mock_search.invoke.side_effect = Exception("API down")

        with patch("langchain_tavily.TavilySearch", return_value=mock_search):
            result = service_with_tavily._gather_news(["AAPL"])
            assert "failed" in result["AAPL"]["summary"].lower()
            assert result["AAPL"]["url"] is None


# ---------------------------------------------------------------------------
# Synthesis (prompt | llm | parser chain)
# ---------------------------------------------------------------------------

class TestSynthesize:
    def test_returns_briefing_result(self):
        """The split chain (prompt | llm → parse) returns a BriefingResult
        with both the structured analysis and extracted thinking.

        Uses LangChain's FakeListChatModel so the chain runs end-to-end.
        """
        from langchain_core.language_models import FakeListChatModel

        fake_llm = FakeListChatModel(responses=[SAMPLE_JSON])
        service = BriefingService(fake_llm)

        result = service._synthesize(
            [{"ticker": "AAPL", "price": 180.0}],
            {"trend": "bull", "volatility": "low", "phase": "markup"},
            {"AAPL": {"summary": "Apple beat earnings", "url": "https://example.com/aapl"}},
        )
        assert isinstance(result, BriefingResult)
        assert isinstance(result.analysis, DailyBriefingAnalysis)
        assert result.analysis.tickers[0].ticker == "AAPL"
        assert result.analysis.tickers[0].news_url == "https://example.com/aapl"
        assert result.analysis.market_regime == SAMPLE_ANALYSIS.market_regime
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
                with patch.object(service, "_gather_news", return_value={"AAPL": {"summary": "news", "url": None}}):
                    with patch.object(service, "_synthesize", return_value=sample_result):
                        service.generate(["AAPL"])
                        result = service.generate(["AAPL"])
                        # _synthesize should only be called once (second call hits cache)
                        assert service._synthesize.call_count == 1
                        assert isinstance(result, BriefingResult)
                        assert result.thinking == "some thought"

    def test_different_tickers_not_cached(self, service, mock_llm):
        sample_result = BriefingResult(analysis=SAMPLE_ANALYSIS, thinking="")
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "X", "price": 1}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                with patch.object(service, "_gather_news", return_value={"X": {"summary": "news", "url": None}}):
                    with patch.object(service, "_synthesize", return_value=sample_result):
                        service.generate(["AAPL"])
                        service.generate(["MSFT"])
                        assert service._synthesize.call_count == 2
