"""Chart data endpoint — serves OHLCV candles + indicator time series."""

import re
from enum import Enum
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()


class ChartPeriod(str, Enum):
    one_week = "1w"
    one_month = "1mo"
    three_months = "3mo"
    six_months = "6mo"
    one_year = "1y"


# Map our period values to yfinance period strings
_PERIOD_MAP = {
    "1w": "5d",
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
}

# All available chart indicators
_ALL_INDICATORS = {"ma5", "ma10", "ma20", "ma50", "ma200", "rsi", "macd", "bollinger"}


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

    from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
    from agents.technical_workflow.process_technical_indicators import TechnicalIndicators

    retriever = YahooFinanceDataRetrieval(ticker)
    yf_period = _PERIOD_MAP[period.value]
    df = retriever.get_historical_prices(period=yf_period)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    # Format candles
    candles = []
    for i in range(len(df)):
        row = df.iloc[i]
        candles.append({
            "time": df.index[i].strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    # Calculate indicators
    ti = TechnicalIndicators(ticker)
    all_chart_indicators = ti.calculate_chart_indicators(df)

    filtered_indicators = {
        k: v for k, v in all_chart_indicators.items() if k in requested
    }

    return JSONResponse(
        content={"candles": candles, "indicators": filtered_indicators},
        headers={"Cache-Control": "public, max-age=60"},
    )
