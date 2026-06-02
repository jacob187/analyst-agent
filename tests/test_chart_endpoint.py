"""Tests for the /stock/{ticker}/chart REST endpoint.

Mocks yfinance so tests run without network access or API keys.
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


@pytest.fixture(autouse=True)
def _clear_retriever_cache():
    """Clear the indicator-window retriever cache so mocks don't leak across tests."""
    from agents.technical_workflow import indicator_window

    indicator_window._retriever_cache.clear()
    yield
    indicator_window._retriever_cache.clear()


@pytest.fixture
def mock_yfinance_data():
    """Build a realistic OHLCV DataFrame that YahooFinanceDataRetrieval would return."""
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
def mock_retriever(mock_yfinance_data):
    """Patch YahooFinanceDataRetrieval at its source module."""
    with patch(
        "agents.technical_workflow.indicator_window.YahooFinanceDataRetrieval"
    ) as MockClass:
        instance = MagicMock()
        instance.get_historical_prices.return_value = mock_yfinance_data
        instance.get_live_price.return_value = {
            "price": 150.00,
            "previousClose": 149.50,
            "change": 0.50,
            "changePercent": 0.33,
        }
        MockClass.return_value = instance
        yield instance


# ── Happy path ─────────────────────────────────────────────────────────────


class TestChartEndpointSuccess:
    def test_returns_candles_and_indicators(self, client, mock_retriever):
        resp = client.get("/stock/AAPL/chart")
        assert resp.status_code == 200
        data = resp.json()
        assert "candles" in data
        assert "indicators" in data

    def test_candles_format(self, client, mock_retriever):
        data = client.get("/stock/AAPL/chart").json()
        candle = data["candles"][0]
        assert set(candle.keys()) == {"time", "open", "high", "low", "close", "volume"}
        assert isinstance(candle["time"], str)
        assert len(candle["time"]) == 10  # YYYY-MM-DD
        assert isinstance(candle["volume"], int)

    def test_candles_count_matches_data(self, client, mock_retriever):
        data = client.get("/stock/AAPL/chart").json()
        assert len(data["candles"]) == 252

    def test_default_indicators_present(self, client, mock_retriever):
        data = client.get("/stock/AAPL/chart").json()
        indicators = data["indicators"]
        assert "ma20" in indicators
        assert "ma50" in indicators
        assert "ma200" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger" in indicators

    def test_indicator_series_are_lists(self, client, mock_retriever):
        data = client.get("/stock/AAPL/chart").json()
        for key in ["ma20", "rsi"]:
            assert isinstance(data["indicators"][key], list)
            assert len(data["indicators"][key]) > 0

    def test_cache_control_header(self, client, mock_retriever):
        resp = client.get("/stock/AAPL/chart")
        assert "max-age=60" in resp.headers.get("cache-control", "")


# ── Period parameter ───────────────────────────────────────────────────────


class TestChartPeriods:
    @pytest.mark.parametrize("period", ["1w", "1mo", "3mo", "6mo", "1y"])
    def test_valid_periods_accepted(self, client, mock_retriever, period):
        resp = client.get(f"/stock/AAPL/chart?period={period}")
        assert resp.status_code == 200

    def test_invalid_period_rejected(self, client, mock_retriever):
        resp = client.get("/stock/AAPL/chart?period=5y")
        assert resp.status_code == 422


# ── Indicator filtering ───────────────────────────────────────────────────


class TestChartIndicatorFiltering:
    def test_request_single_indicator(self, client, mock_retriever):
        data = client.get("/stock/AAPL/chart?indicators=rsi").json()
        assert "rsi" in data["indicators"]
        # Other indicators should be absent
        assert "macd" not in data["indicators"]
        assert "ma20" not in data["indicators"]

    def test_request_multiple_indicators(self, client, mock_retriever):
        data = client.get("/stock/AAPL/chart?indicators=rsi,macd").json()
        assert "rsi" in data["indicators"]
        assert "macd" in data["indicators"]
        assert "bollinger" not in data["indicators"]

    def test_unknown_indicator_rejected(self, client, mock_retriever):
        resp = client.get("/stock/AAPL/chart?indicators=rsi,fake_indicator")
        assert resp.status_code == 422
        assert "fake_indicator" in resp.json()["detail"]


# ── Error cases ────────────────────────────────────────────────────────────


class TestChartEndpointErrors:
    def test_invalid_ticker_format(self, client):
        resp = client.get("/stock/INVALID!!!/chart")
        assert resp.status_code == 422

    def test_no_data_returns_404(self, client):
        with patch("agents.technical_workflow.indicator_window.YahooFinanceDataRetrieval") as MockClass:
            instance = MagicMock()
            instance.get_historical_prices.return_value = None
            MockClass.return_value = instance
            resp = client.get("/stock/ZZZZ/chart")
            assert resp.status_code == 404
            assert "No data" in resp.json()["detail"]

    def test_empty_dataframe_returns_404(self, client):
        with patch("agents.technical_workflow.indicator_window.YahooFinanceDataRetrieval") as MockClass:
            instance = MagicMock()
            instance.get_historical_prices.return_value = pd.DataFrame()
            MockClass.return_value = instance
            resp = client.get("/stock/ZZZZ/chart")
            assert resp.status_code == 404

    def test_ticker_case_insensitive(self, client, mock_retriever):
        """Lowercase ticker should work and be uppercased internally."""
        resp = client.get("/stock/aapl/chart")
        assert resp.status_code == 200


# ── Bounded caches (Phase 3 hardening) ─────────────────────────────────────


class TestChartCacheBounded:
    """The module-level chart caches must evict under unique-key rotation
    so memory cannot grow unbounded. Mirrors tests/test_sec_tools_caches.py."""

    def setup_method(self):
        from api.routes.chart import _chart_cache, _quotes_cache
        _chart_cache.clear()
        _quotes_cache.clear()

    def test_chart_cache_evicts_at_maxsize(self):
        from cachetools import TTLCache
        from api.routes import chart

        assert isinstance(chart._chart_cache, TTLCache)
        for i in range(1000):
            chart._chart_cache[(f"T{i}", "1y")] = {"candles": []}
        assert len(chart._chart_cache) <= 256

    def test_quotes_cache_evicts_at_maxsize(self):
        from cachetools import TTLCache
        from api.routes import chart

        assert isinstance(chart._quotes_cache, TTLCache)
        for i in range(1000):
            chart._quotes_cache[frozenset([f"T{i}"])] = {}
        assert len(chart._quotes_cache) <= 256


class TestChartIndicatorValidation:
    """The `indicators` query param is bounded so a 1MB string can't reach
    the parser. Defends against a cheap DoS that would otherwise allocate
    a huge split list per request."""

    def test_oversize_indicators_string_returns_422(self, client, mock_retriever):
        oversize = "x," * 1000  # ~2000 chars, > max_length=200
        resp = client.get(f"/stock/AAPL/chart?indicators={oversize}")
        assert resp.status_code == 422

    def test_under_limit_indicators_string_accepted(self, client, mock_retriever):
        resp = client.get("/stock/AAPL/chart?indicators=rsi,macd")
        assert resp.status_code == 200
