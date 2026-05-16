"""Indicator-aware market data fetching.

Long-window technical indicators (MA200, MA50, MACD) are only valid after
their pre-roll period. Fetching exactly the user's display window means
MA200 at the right edge of the window was computed on a stub of data — it
lags real price by ~50 bars and is effectively useless for the bulk of the
window.

This module owns the math for "fetch enough bars that indicators are valid
through the display window, then trim back for rendering." Both the chart
REST endpoint and the LLM technical-analysis tools go through here so the
pre-roll rules live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from cachetools import LRUCache

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


# (display_period, interval) → yfinance fetch_period.
# Sized so MA200 has enough lookback to be populated through every bar
# of the display window. yfinance caps: 15m → 60d, 1h → 730d.
_FETCH_PERIOD: dict[tuple[str, str], str] = {
    ("5d",  "15m"): "60d",
    ("1mo", "1h"):  "6mo",
    ("3mo", "1h"):  "9mo",
    ("6mo", "1d"):  "2y",
    ("1y",  "1d"):  "2y",
}

# How far back from the latest bar the display window extends.
_DISPLAY_DELTA: dict[str, pd.Timedelta] = {
    "5d":  pd.Timedelta(days=7),
    "1mo": pd.Timedelta(days=31),
    "3mo": pd.Timedelta(days=93),
    "6mo": pd.Timedelta(days=186),
    "1y":  pd.Timedelta(days=366),
}


# Process-wide retriever cache so the chart route and LLM tools don't each
# spin up their own yfinance.Ticker instances for the same symbol.
_retriever_cache: LRUCache = LRUCache(maxsize=128)


def get_retriever(ticker: str) -> YahooFinanceDataRetrieval:
    if ticker not in _retriever_cache:
        _retriever_cache[ticker] = YahooFinanceDataRetrieval(ticker)
    return _retriever_cache[ticker]


@dataclass
class IndicatorWindow:
    """Two views of one fetch.

    `full`:    extended bars — pass to indicator computation so MA200 etc.
               are populated through every bar in `display`.
    `display`: the user's actual visible window — use for chart rendering,
               pattern detection, and anywhere extra pre-roll would mislead.
    """
    full: pd.DataFrame
    display: pd.DataFrame


def fetch_indicator_window(
    ticker: str, display_period: str, interval: str = "1d"
) -> Optional[IndicatorWindow]:
    """Fetch enough bars that long-window indicators are valid across the display.

    Args:
        ticker: stock symbol
        display_period: the user's window — one of "5d", "1mo", "3mo", "6mo", "1y"
        interval: yfinance bar interval — "1d", "1h", or "15m"

    Returns IndicatorWindow with both views, or None if no data is available.
    Unknown (display_period, interval) combinations fall back to fetching the
    display period directly with no pre-roll.
    """
    fetch_period = _FETCH_PERIOD.get((display_period, interval), display_period)

    retriever = get_retriever(ticker)
    full = retriever.get_historical_prices(period=fetch_period, interval=interval)
    if full is None or full.empty:
        return None

    delta = _DISPLAY_DELTA.get(display_period)
    if delta is None:
        display = full
    else:
        cutoff = full.index.max() - delta
        display = full.loc[full.index >= cutoff]

    return IndicatorWindow(full=full, display=display)
