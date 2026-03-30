"""Tests for BriefingService."""

from unittest.mock import MagicMock, patch
import pytest
import agents.briefing.briefing_service as briefing_mod
from agents.briefing.briefing_service import BriefingService


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Morning briefing: Markets are bullish.")
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


class TestFormatMarketContext:
    def test_with_full_data(self, service):
        regime = {
            "trend": {"type": "bull", "price": 500.0, "ma50": 490.0, "ma200": 470.0},
            "volatility": {"regime": "low", "current_vix": 14.5, "description": "Calm"},
            "phase": {"phase": "markup", "description": "Uptrend"},
            "recommendations": ["Buy dips", "Use trend-following"],
        }
        result = service._format_market_context(regime)
        assert "BULL" in result
        assert "14.5" in result

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


class TestSynthesize:
    def test_calls_llm(self, service, mock_llm):
        result = service._synthesize(
            [{"ticker": "AAPL", "price": 180.0}],
            {
                "trend": {"type": "bull", "price": 500, "ma50": 490, "ma200": 470},
                "volatility": {"regime": "low", "current_vix": 14, "description": "Calm"},
                "phase": {"phase": "markup", "description": "Up"},
                "recommendations": [],
            },
        )
        mock_llm.invoke.assert_called_once()
        assert "briefing" in result.lower()

    def test_handles_list_content(self, service):
        """Gemini with thinking returns content as list of dicts."""
        service.llm.invoke.return_value = MagicMock(
            content=[{"type": "text", "text": "Briefing text here"}]
        )
        result = service._synthesize([{"ticker": "X", "price": 1}], {"error": "no data"})
        assert "Briefing text here" in result


class TestCaching:
    def test_second_call_cached(self, service, mock_llm):
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "AAPL", "price": 180}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                service.generate(["AAPL"])
                service.generate(["AAPL"])
                # LLM should only be called once
                assert mock_llm.invoke.call_count == 1

    def test_different_tickers_not_cached(self, service, mock_llm):
        with patch.object(service, "_gather_ticker_data", return_value=[{"ticker": "X", "price": 1}]):
            with patch.object(service, "_get_market_regime", return_value={"error": "skip"}):
                service.generate(["AAPL"])
                service.generate(["MSFT"])
                assert mock_llm.invoke.call_count == 2
