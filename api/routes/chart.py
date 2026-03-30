"""Chart data endpoint — serves OHLCV candles + indicator time series.

Performance notes:
- Server-side TTL cache avoids repeated yfinance network calls (200-800ms each).
  The cache stores the full payload (candles + ALL indicators); per-request
  indicator filtering is a cheap dict comprehension after the cache lookup.
- Candle formatting uses vectorized pandas ops instead of row-by-row iloc.
"""

import re
import time
from enum import Enum
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()


class ChartPeriod(str, Enum):
    one_week = "1w"
    one_month = "1mo"
    three_months = "3mo"
    six_months = "6mo"
    one_year = "1y"


# Map our period values to yfinance (period, interval) pairs.
# Short timeframes use intraday intervals for more useful candle density.
_PERIOD_MAP: dict[str, tuple[str, str]] = {
    "1w": ("5d", "15m"),     # ~130 candles (15-min bars)
    "1mo": ("1mo", "1h"),    # ~140 candles (hourly bars)
    "3mo": ("3mo", "1h"),    # ~130 candles (hourly bars)
    "6mo": ("6mo", "1d"),    # ~126 candles (daily bars)
    "1y": ("1y", "1d"),      # ~252 candles (daily bars)
}

# All available chart indicators
_ALL_INDICATORS = {"ma5", "ma10", "ma20", "ma50", "ma200", "rsi", "macd", "bollinger"}

# Server-side cache: {(ticker, period): (monotonic_timestamp, full_payload)}
# Stores candles + all indicators so different indicator requests share one fetch.
_chart_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_CACHE_TTL = 60  # seconds


def _format_candles(df: pd.DataFrame, intraday: bool = False) -> list[dict]:
    """Vectorized OHLCV formatting.

    For daily data, uses "YYYY-MM-DD" strings (Lightweight Charts date format).
    For intraday data, uses Unix timestamps (Lightweight Charts UTCTimestamp).
    """
    if intraday:
        # Convert timezone-aware datetimes to UTC unix timestamps
        times = (df.index.astype("int64") // 10**9).tolist()
    else:
        times = df.index.strftime("%Y-%m-%d").tolist()

    candle_df = pd.DataFrame({
        "time": times,
        "open": df["Open"].round(2),
        "high": df["High"].round(2),
        "low": df["Low"].round(2),
        "close": df["Close"].round(2),
        "volume": df["Volume"].astype(int),
    })
    return candle_df.to_dict("records")


@router.get("/stock/{ticker}/chart")
async def get_chart_data(
    ticker: str,
    period: ChartPeriod = ChartPeriod.one_year,
    indicators: Optional[str] = Query(
        default="ma20,ma50,ma200,rsi,macd,bollinger",
        description="Comma-separated indicator names",
    ),
):
    """Return OHLCV candles and indicator time series for chart rendering."""
    if not re.match(r"^[A-Za-z0-9.\-]{1,10}$", ticker):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    ticker = ticker.upper()

    # Parse requested indicators
    requested = set(indicators.split(",")) if indicators else _ALL_INDICATORS
    invalid = requested - _ALL_INDICATORS
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown indicators: {', '.join(invalid)}. Valid: {', '.join(sorted(_ALL_INDICATORS))}",
        )

    # Check server-side cache (keyed on ticker + period, not indicators —
    # we cache the full payload and filter afterward)
    cache_key = (ticker, period.value)
    now = time.monotonic()
    cached = _chart_cache.get(cache_key)

    if cached and (now - cached[0]) < _CACHE_TTL:
        full_data = cached[1]
    else:
        from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
        from agents.technical_workflow.process_technical_indicators import TechnicalIndicators

        retriever = YahooFinanceDataRetrieval(ticker)
        yf_period, yf_interval = _PERIOD_MAP[period.value]
        df = retriever.get_historical_prices(period=yf_period, interval=yf_interval)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        intraday = yf_interval != "1d"
        candles = _format_candles(df, intraday=intraday)

        ti = TechnicalIndicators(ticker)
        all_indicators = ti.calculate_chart_indicators(df, intraday=intraday)


        # Pattern detection
        patterns: list[dict] = []
        try:
            from agents.technical_workflow.pattern_recognition import PatternRecognitionEngine
            engine = PatternRecognitionEngine()
            raw_patterns = engine.detect_all_patterns(df)
            for p in raw_patterns:
                pattern_entry: dict = {
                    "type": p.get("type", ""),
                    "direction": p.get("direction", ""),
                    "confidence": round(p.get("confidence", 0), 2),
                }
                # For patterns with a specific bar (crossovers), include the time
                if "days_ago" in p and candles:
                    idx = max(0, len(candles) - 1 - p["days_ago"])
                    pattern_entry["time"] = candles[idx]["time"]
                patterns.append(pattern_entry)
        except Exception:
            pass  # Patterns are supplementary

        # Live quote from yfinance fast_info (lightweight, ~50ms)
        quote = {}
        try:
            fi = retriever.yf_ticker.fast_info
            prev_close = float(fi["previous_close"])
            last_price = float(fi["last_price"])
            quote = {
                "price": round(last_price, 2),
                "previousClose": round(prev_close, 2),
                "change": round(last_price - prev_close, 2),
                "changePercent": round((last_price - prev_close) / prev_close * 100, 2),
            }
        except Exception:
            pass  # Fall back to candle data on frontend

        full_data = {
            "candles": candles,
            "all_indicators": all_indicators,
            "quote": quote,
            "patterns": patterns,
        }
        _chart_cache[cache_key] = (now, full_data)

    # Filter to only requested indicators (cheap dict comprehension)
    filtered_indicators = {
        k: v for k, v in full_data["all_indicators"].items() if k in requested
    }

    return JSONResponse(
        content={
            "candles": full_data["candles"],
            "indicators": filtered_indicators,
            "quote": full_data.get("quote", {}),
            "patterns": full_data.get("patterns", []),
        },
        headers={"Cache-Control": "public, max-age=60"},
    )
