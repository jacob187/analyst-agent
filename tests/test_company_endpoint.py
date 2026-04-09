"""Tests for the /api/company/{ticker}/profile endpoint.

Mocks yfinance and technical analysis so tests run without network access.
Follows the same TestClient + patch pattern as test_chart_endpoint.py.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_ohlcv():
    """Realistic 1-year OHLCV DataFrame for technical indicator calculation."""
    rng = np.random.default_rng(42)
    rows = 252
    dates = pd.bdate_range(end="2026-03-27", periods=rows)
    returns = rng.normal(0.0005, 0.02, size=rows)
    close = 150 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.uniform(-0.01, 0.01, size=rows)),
            "High": close * (1 + rng.uniform(0, 0.02, size=rows)),
            "Low": close * (1 - rng.uniform(0, 0.02, size=rows)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, size=rows),
        },
        index=dates,
    )


@pytest.fixture
def mock_profile_deps(mock_ohlcv):
    """Patch all external dependencies used by the profile endpoint."""
    # Clear the profile cache so each test starts fresh
    from api.routes.company import _profile_cache
    _profile_cache.clear()

    with (
        patch("agents.technical_workflow.get_stock_data.YahooFinanceDataRetrieval") as MockRetriever,
        patch("agents.technical_workflow.process_technical_indicators.TechnicalIndicators") as MockTI,
        patch("agents.technical_workflow.pattern_recognition.PatternRecognitionEngine") as MockPattern,
        patch("agents.market_analysis.regime_detector.MarketRegimeDetector") as MockRegime,
    ):
        # YahooFinanceDataRetrieval instance
        retriever = MagicMock()
        retriever.get_company_profile.return_value = {
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "website": "https://www.apple.com",
            "longBusinessSummary": "Apple designs consumer electronics.",
            "fullTimeEmployees": 164000,
            "marketCap": 3800000000000,
            "trailingPE": 33.5,
            "forwardPE": 28.2,
            "priceToBook": 55.1,
            "fiftyTwoWeekHigh": 260.10,
            "fiftyTwoWeekLow": 164.08,
            "dividendYield": 0.0044,
            "beta": 1.24,
        }
        retriever.get_live_price.return_value = {
            "price": 246.63,
            "previousClose": 246.01,
            "change": 0.62,
            "changePercent": 0.25,
        }
        retriever.get_earnings_calendar.return_value = {
            "earnings_date": "2026-04-24",
            "earnings_average": 2.35,
            "revenue_average": 94500000000,
        }
        retriever.get_historical_prices.return_value = mock_ohlcv
        MockRetriever.return_value = retriever

        # TechnicalIndicators instance
        ti_instance = MagicMock()
        ti_instance.calculate_all_indicators.return_value = {
            "rsi": {"current": 58.3, "signal": "neutral"},
            "macd": {"macd_line": 1.2, "signal_line": 0.8, "histogram": 0.4, "signal": "bullish"},
            "adx": {"adx": 24.5, "plus_di": 28.1, "minus_di": 18.3, "trend_strength": "developing"},
            "bollinger_bands": {"upper_band": 255.0, "middle_band": 245.0, "lower_band": 235.0, "position": "within_bands"},
            "moving_averages": {"MA_20": 244.5, "MA_50": 240.0, "MA_200": 220.0, "latest_close": 246.63},
            "volatility": {"daily_volatility": 0.018, "annualized_volatility": 0.285},
        }
        MockTI.return_value = ti_instance

        # PatternRecognitionEngine instance
        pattern_instance = MagicMock()
        pattern_instance.detect_all_patterns.return_value = [
            {"type": "golden_cross", "direction": "bullish", "confidence": 0.85, "status": "confirmed"},
        ]
        MockPattern.return_value = pattern_instance

        # MarketRegimeDetector instance
        regime_instance = MagicMock()
        regime_instance.detect_regime.return_value = {
            "trend": "bull",
            "volatility": "moderate",
            "phase": "markup",
            "recommendations": ["Favor long positions"],
        }
        MockRegime.return_value = regime_instance

        yield {
            "retriever": retriever,
            "ti": ti_instance,
            "pattern": pattern_instance,
            "regime": regime_instance,
        }


# ── Happy path ─────────────────────────────────────────────────────────────


class TestProfileEndpointSuccess:
    def test_returns_200_with_expected_top_level_keys(self, client, mock_profile_deps):
        resp = client.get("/api/company/AAPL/profile")
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = {"ticker", "company", "metrics", "quote", "earnings", "technicals", "patterns", "regime"}
        assert set(data.keys()) == expected_keys

    def test_ticker_is_uppercased(self, client, mock_profile_deps):
        data = client.get("/api/company/aapl/profile").json()
        assert data["ticker"] == "AAPL"

    def test_company_metadata_shape(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        company = data["company"]
        assert company["name"] == "Apple Inc."
        assert company["sector"] == "Technology"
        assert company["industry"] == "Consumer Electronics"
        assert company["summary"] is not None
        assert company["employees"] == 164000

    def test_metrics_shape(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        metrics = data["metrics"]
        assert metrics["market_cap"] == 3800000000000
        assert metrics["pe_ratio"] == 33.5
        assert metrics["forward_pe"] == 28.2
        assert isinstance(metrics["52wk_high"], (int, float))
        assert isinstance(metrics["beta"], (int, float))

    def test_quote_shape(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        quote = data["quote"]
        assert quote["price"] == 246.63
        assert "change" in quote
        assert "changePercent" in quote

    def test_earnings_shape(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        earnings = data["earnings"]
        assert earnings["earnings_date"] == "2026-04-24"
        assert isinstance(earnings["earnings_average"], (int, float))

    def test_technicals_contains_expected_indicators(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        technicals = data["technicals"]
        assert "rsi" in technicals
        assert "macd" in technicals
        assert "adx" in technicals
        assert technicals["rsi"]["signal"] == "neutral"
        assert technicals["macd"]["signal"] == "bullish"

    def test_patterns_shape(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        patterns = data["patterns"]
        assert len(patterns) == 1
        assert patterns[0]["type"] == "golden_cross"
        assert patterns[0]["direction"] == "bullish"
        assert isinstance(patterns[0]["confidence"], float)

    def test_regime_shape(self, client, mock_profile_deps):
        data = client.get("/api/company/AAPL/profile").json()
        regime = data["regime"]
        assert regime["trend"] == "bull"
        assert regime["volatility"] == "moderate"
        assert regime["phase"] == "markup"


# ── Caching ────────────────────────────────────────────────────────────────


class TestProfileCaching:
    def test_cache_control_header(self, client, mock_profile_deps):
        resp = client.get("/api/company/AAPL/profile")
        assert "max-age=300" in resp.headers.get("cache-control", "")

    def test_cache_hit_does_not_refetch(self, client, mock_profile_deps):
        """Second request within TTL should serve from cache, not call retriever again."""
        client.get("/api/company/AAPL/profile")
        client.get("/api/company/AAPL/profile")
        # get_company_profile called only once — second request served from cache
        assert mock_profile_deps["retriever"].get_company_profile.call_count == 1


# ── Error cases ────────────────────────────────────────────────────────────


class TestProfileEndpointErrors:
    def test_invalid_ticker_format(self, client):
        resp = client.get("/api/company/INVALID!!!/profile")
        assert resp.status_code == 422

    def test_ticker_with_dot_is_valid(self, client, mock_profile_deps):
        """Tickers like BRK.B should be accepted."""
        resp = client.get("/api/company/BRK.B/profile")
        assert resp.status_code == 200


# ── Partial failure resilience ─────────────────────────────────────────────


class TestProfilePartialFailure:
    def test_returns_data_even_if_patterns_fail(self, client, mock_profile_deps):
        """If pattern detection throws, the rest of the profile should still return."""
        mock_profile_deps["pattern"].detect_all_patterns.side_effect = RuntimeError("boom")
        resp = client.get("/api/company/MSFT/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patterns"] == []
        assert data["technicals"]["rsi"]["signal"] == "neutral"

    def test_returns_data_even_if_regime_fails(self, client, mock_profile_deps):
        mock_profile_deps["regime"].detect_regime.side_effect = RuntimeError("boom")
        resp = client.get("/api/company/MSFT/profile")
        assert resp.status_code == 200
        assert resp.json()["regime"] == {}

    def test_returns_empty_technicals_if_no_price_data(self, client, mock_profile_deps):
        mock_profile_deps["retriever"].get_historical_prices.return_value = None
        resp = client.get("/api/company/NEWIPO/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["technicals"] == {}
        assert data["patterns"] == []
        # Company info should still be present
        assert data["company"]["name"] == "Apple Inc."
