"""Tests for indicator_window — the centralized indicator-aware fetch module.

Uses a real fake retriever (not MagicMock) per the project rule:
external APIs are stubbed with fixture DataFrames that mirror real shapes;
internal classes are exercised with real instances.
"""

from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agents.technical_workflow import indicator_window
from agents.technical_workflow.indicator_window import (
    IndicatorWindow,
    fetch_indicator_window,
    get_retriever,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


def _build_ohlcv_fixture(rows: int = 504, end: str = "2026-05-15") -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame matching yfinance's output shape."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=end, periods=rows)
    close = 150 * np.cumprod(1 + rng.normal(0.0005, 0.02, size=rows))
    return pd.DataFrame(
        {
            "Open":   close * (1 + rng.uniform(-0.01, 0.01, size=rows)),
            "High":   close * (1 + rng.uniform(0, 0.02, size=rows)),
            "Low":    close * (1 - rng.uniform(0, 0.02, size=rows)),
            "Close":  close,
            "Volume": rng.integers(1_000_000, 10_000_000, size=rows),
        },
        index=dates,
    )


_TWO_YEAR_DAILY = _build_ohlcv_fixture(rows=504)


class FakeYFinanceRetriever:
    """Real Python class that mimics YahooFinanceDataRetrieval at the boundary.

    Returns the fixture DataFrame from `get_historical_prices` and records the
    call arguments so tests can assert on what was requested.
    """

    # Class-level instance registry so tests can introspect what was created.
    instances: list["FakeYFinanceRetriever"] = []

    def __init__(self, ticker: str, df: Optional[pd.DataFrame] = None):
        self.ticker = ticker
        self._df = df if df is not None else _TWO_YEAR_DAILY
        self.calls: list[dict] = []
        FakeYFinanceRetriever.instances.append(self)

    def get_historical_prices(self, period: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        self.calls.append({"period": period, "interval": interval})
        return self._df

    def get_live_price(self) -> dict:
        return {"price": 150.00, "previousClose": 149.50}


@pytest.fixture(autouse=True)
def _clear_retriever_cache_and_registry():
    """Reset module-level state so tests don't leak through each other."""
    indicator_window._retriever_cache.clear()
    FakeYFinanceRetriever.instances.clear()
    yield
    indicator_window._retriever_cache.clear()
    FakeYFinanceRetriever.instances.clear()


@pytest.fixture
def fake_retriever_class():
    """Substitute the real retriever class with the fake at the import site.

    This is the only patch we need: the boundary where indicator_window reaches
    out to yfinance. Everything inside (the cache, the trim math) runs real."""
    with patch(
        "agents.technical_workflow.indicator_window.YahooFinanceDataRetrieval",
        FakeYFinanceRetriever,
    ):
        yield FakeYFinanceRetriever


@pytest.fixture
def fake_retriever_returning_none():
    """Boundary stub for the no-data case."""

    class _NoDataRetriever(FakeYFinanceRetriever):
        def get_historical_prices(self, period, interval="1d"):
            return None

    with patch(
        "agents.technical_workflow.indicator_window.YahooFinanceDataRetrieval",
        _NoDataRetriever,
    ):
        yield


@pytest.fixture
def fake_retriever_returning_empty():
    """Boundary stub for the empty-DataFrame case."""

    class _EmptyRetriever(FakeYFinanceRetriever):
        def get_historical_prices(self, period, interval="1d"):
            return pd.DataFrame()

    with patch(
        "agents.technical_workflow.indicator_window.YahooFinanceDataRetrieval",
        _EmptyRetriever,
    ):
        yield


# ── Tests ──────────────────────────────────────────────────────────────────


class TestFetchIndicatorWindow:
    def test_returns_indicator_window(self, fake_retriever_class):
        result = fetch_indicator_window("AAPL", "1y", "1d")
        assert isinstance(result, IndicatorWindow)

    def test_full_has_two_years_of_bars(self, fake_retriever_class):
        result = fetch_indicator_window("AAPL", "1y", "1d")
        assert len(result.full) == 504  # matches the 2-year fixture

    def test_display_is_trimmed_to_one_year(self, fake_retriever_class):
        result = fetch_indicator_window("AAPL", "1y", "1d")
        # 1y delta is 366 calendar days → ~252-265 business days
        assert 240 <= len(result.display) <= 270
        assert len(result.display) < len(result.full)

    def test_display_is_tail_slice_of_full(self, fake_retriever_class):
        result = fetch_indicator_window("AAPL", "1y", "1d")
        assert result.display.index[-1] == result.full.index[-1]
        assert result.display.index[0] >= result.full.index[0]

    def test_six_month_display_smaller_than_one_year(self, fake_retriever_class):
        one_year = fetch_indicator_window("AAPL", "1y", "1d")
        six_month = fetch_indicator_window("AAPL", "6mo", "1d")
        assert len(six_month.display) < len(one_year.display)

    def test_one_year_display_requests_two_years_from_yfinance(self, fake_retriever_class):
        fetch_indicator_window("AAPL", "1y", "1d")
        retriever = FakeYFinanceRetriever.instances[-1]
        assert retriever.calls == [{"period": "2y", "interval": "1d"}]

    def test_unknown_period_falls_back_to_display_period(self, fake_retriever_class):
        """Unmapped (display, interval) combos use display_period as fetch_period."""
        fetch_indicator_window("AAPL", "max", "1d")
        retriever = FakeYFinanceRetriever.instances[-1]
        assert retriever.calls == [{"period": "max", "interval": "1d"}]

    def test_returns_none_when_no_data(self, fake_retriever_returning_none):
        assert fetch_indicator_window("ZZZZ", "1y", "1d") is None

    def test_returns_none_when_empty_dataframe(self, fake_retriever_returning_empty):
        assert fetch_indicator_window("ZZZZ", "1y", "1d") is None


class TestRetrieverCache:
    def test_caches_retriever_per_ticker(self, fake_retriever_class):
        a1 = get_retriever("AAPL")
        a2 = get_retriever("AAPL")
        assert a1 is a2
        # Only one fake should have been constructed for AAPL.
        aapl_instances = [r for r in FakeYFinanceRetriever.instances if r.ticker == "AAPL"]
        assert len(aapl_instances) == 1

    def test_different_tickers_get_different_retrievers(self, fake_retriever_class):
        a = get_retriever("AAPL")
        m = get_retriever("MSFT")
        assert a is not m
        assert a.ticker == "AAPL"
        assert m.ticker == "MSFT"


class TestMA200LookbackIsValid:
    """Behavioral test: after centralization, MA200 should be valid through
    every bar of the display window — not just the last ~50."""

    def test_ma200_populated_across_display_window(self, fake_retriever_class):
        from agents.technical_workflow.process_technical_indicators import (
            TechnicalIndicators,
        )

        result = fetch_indicator_window("AAPL", "1y", "1d")
        assert result is not None

        ti = TechnicalIndicators("AAPL")
        raw = ti._calculate_all_raw(result.full)
        ma200 = raw["moving_averages"]["series"].get("MA_200")
        assert ma200 is not None

        # MA200 at the first bar of the display window must be a real number.
        # If we'd computed on display alone, this would be NaN.
        display_start = result.display.index[0]
        assert not pd.isna(ma200.loc[display_start])
