"""Tests for YahooFinanceDataRetrieval.get_historical_prices."""

import math

import pandas as pd

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


class _FakeYfTicker:
    """Stands in for yf.Ticker at the external-API boundary."""

    def __init__(self, hist: pd.DataFrame):
        self._hist = hist

    def history(self, period: str, interval: str) -> pd.DataFrame:
        return self._hist


def _ohlcv(rows: list[tuple[float, float, float, float, int]]) -> pd.DataFrame:
    index = pd.date_range("2026-10-05", periods=len(rows), freq="D", tz="America/New_York")
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close", "Volume"])


def _retriever(hist: pd.DataFrame) -> YahooFinanceDataRetrieval:
    r = YahooFinanceDataRetrieval("NVDA")
    r.yf_ticker = _FakeYfTicker(hist)
    return r


def test_drops_placeholder_bar_with_nan_ohlc():
    # Mirrors yfinance after the close: today's bar exists but OHLC are NaN.
    hist = _ohlcv([
        (100.0, 102.0, 99.0, 101.0, 1_000),
        (101.0, 103.0, 100.0, 102.0, 1_200),
        (math.nan, math.nan, math.nan, math.nan, 0),
    ])

    result = _retriever(hist).get_historical_prices(period="5d")

    assert len(result) == 2
    assert not result[["Open", "High", "Low", "Close"]].isna().any().any()
    assert result["Close"].iloc[-1] == 102.0


def test_all_nan_bars_returns_none():
    hist = _ohlcv([(math.nan, math.nan, math.nan, math.nan, 0)])

    assert _retriever(hist).get_historical_prices(period="5d") is None
