import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class TechnicalIndicators:
    """Class to calculate technical indicators from stock data.

    Serves two consumers:
    - Agent tools: call calculate_all_indicators() for current scalar values
    - Chart endpoint: call calculate_chart_indicators() for full time series
    Both use the same underlying calculations — single source of truth.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker

    # ── Public API ──────────────────────────────────────────────────────

    def calculate_all_indicators(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate all technical indicators, returning current values only.

        This is the existing interface used by agent tools. Return shape is
        unchanged from the original implementation.
        """
        if df is None or df.empty:
            print(
                f"Warning: No price DataFrame provided for {self.ticker}. Skipping indicator calculation."
            )
            return {}

        raw = self._calculate_all_raw(df)

        return {
            "moving_averages": raw["moving_averages"]["current"],
            "rsi": raw["rsi"]["current"],
            "macd": raw["macd"]["current"],
            "bollinger_bands": raw["bollinger_bands"]["current"],
            "volatility": raw["volatility"],
        }

    def calculate_chart_indicators(
        self, df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate indicators and return full time series for chart rendering.

        Returns dict with keys like 'ma20', 'rsi', 'macd', 'bollinger' —
        each containing a list of {time, value} dicts ready for Lightweight Charts.
        """
        if df is None or df.empty:
            return {}

        raw = self._calculate_all_raw(df)
        index = df.index
        result = {}

        # Moving averages
        ma_series = raw["moving_averages"]["series"]
        for key, label in [("MA_5", "ma5"), ("MA_10", "ma10"), ("MA_20", "ma20"),
                           ("MA_50", "ma50"), ("MA_200", "ma200")]:
            if key in ma_series:
                result[label] = self._series_to_chart_format(ma_series[key], index)

        # RSI
        result["rsi"] = self._series_to_chart_format(
            raw["rsi"]["series"], index
        )

        # MACD — three series merged into one list of dicts
        macd_s = raw["macd"]["series"]
        result["macd"] = self._macd_to_chart_format(
            macd_s["macd_line"], macd_s["signal_line"], macd_s["histogram"], index
        )

        # Bollinger Bands — three bands merged into one list of dicts
        bb_s = raw["bollinger_bands"]["series"]
        result["bollinger"] = self._bollinger_to_chart_format(
            bb_s["upper"], bb_s["middle"], bb_s["lower"], index
        )

        return result

    # ── Internal calculation (shared by both public methods) ────────────

    def _calculate_all_raw(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute all indicators, returning both current values and series."""
        return {
            "moving_averages": self._calculate_moving_averages(df),
            "rsi": self._calculate_rsi(df),
            "macd": self._calculate_macd(df),
            "bollinger_bands": self._calculate_bollinger_bands(df),
            "volatility": self._calculate_volatility(df),
        }

    # ── Chart format helpers ────────────────────────────────────────────

    @staticmethod
    def _series_to_chart_format(
        series: pd.Series, index: pd.DatetimeIndex
    ) -> List[Dict[str, Any]]:
        """Convert a pandas Series to [{time: "YYYY-MM-DD", value: float}].

        Drops NaN entries (from rolling warmup periods).
        """
        result = []
        for i, val in enumerate(series):
            if pd.notna(val) and np.isfinite(val):
                result.append({
                    "time": index[i].strftime("%Y-%m-%d"),
                    "value": round(float(val), 4),
                })
        return result

    @staticmethod
    def _macd_to_chart_format(
        macd_line: pd.Series,
        signal_line: pd.Series,
        histogram: pd.Series,
        index: pd.DatetimeIndex,
    ) -> List[Dict[str, Any]]:
        """Convert MACD series to [{time, macd, signal, histogram}]."""
        result = []
        for i in range(len(macd_line)):
            m, s, h = macd_line.iloc[i], signal_line.iloc[i], histogram.iloc[i]
            if pd.notna(m) and pd.notna(s) and pd.notna(h):
                result.append({
                    "time": index[i].strftime("%Y-%m-%d"),
                    "macd": round(float(m), 4),
                    "signal": round(float(s), 4),
                    "histogram": round(float(h), 4),
                })
        return result

    @staticmethod
    def _bollinger_to_chart_format(
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series,
        index: pd.DatetimeIndex,
    ) -> List[Dict[str, Any]]:
        """Convert Bollinger Band series to [{time, upper, middle, lower}]."""
        result = []
        for i in range(len(upper)):
            u, m, lo = upper.iloc[i], middle.iloc[i], lower.iloc[i]
            if pd.notna(u) and pd.notna(m) and pd.notna(lo):
                result.append({
                    "time": index[i].strftime("%Y-%m-%d"),
                    "upper": round(float(u), 4),
                    "middle": round(float(m), 4),
                    "lower": round(float(lo), 4),
                })
        return result

    # ── Private indicator methods ───────────────────────────────────────
    # Each returns {"current": <scalar dict>, "series": <pandas Series/dict>}

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving averages, returning both current values and full series."""
        close = df["Close"]

        ma_5 = close.rolling(window=5).mean()
        ma_10 = close.rolling(window=10).mean()
        ma_20 = close.rolling(window=20).mean()

        current = {
            "MA_5": ma_5.iloc[-1],
            "MA_10": ma_10.iloc[-1],
            "MA_20": ma_20.iloc[-1],
            "latest_close": close.iloc[-1],
        }
        series = {
            "MA_5": ma_5,
            "MA_10": ma_10,
            "MA_20": ma_20,
        }

        if len(close) >= 50:
            ma_50 = close.rolling(window=50).mean()
            current["MA_50"] = ma_50.iloc[-1]
            series["MA_50"] = ma_50

        if len(close) >= 200:
            ma_200 = close.rolling(window=200).mean()
            current["MA_200"] = ma_200.iloc[-1]
            series["MA_200"] = ma_200
            if "MA_50" in current:
                current["trend_50_200"] = (
                    "bullish" if current["MA_50"] > current["MA_200"] else "bearish"
                )

        return {"current": current, "series": series}

    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> Dict[str, Any]:
        """Calculate RSI, returning both current value and full series."""
        close = df["Close"]
        delta = close.diff()

        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        roll_up = up.ewm(com=periods - 1, adjust=False).mean()
        roll_down = down.ewm(com=periods - 1, adjust=False).mean()

        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        current_rsi = rsi.iloc[-1]

        return {
            "current": {
                "current": current_rsi,
                "signal": (
                    "oversold"
                    if current_rsi < 30
                    else ("overbought" if current_rsi > 70 else "neutral")
                ),
            },
            "series": rsi,
        }

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD, returning both current values and full series."""
        close = df["Close"]

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "current": {
                "macd_line": macd_line.iloc[-1],
                "signal_line": signal_line.iloc[-1],
                "histogram": histogram.iloc[-1],
                "signal": (
                    "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"
                ),
            },
            "series": {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
            },
        }

    def _calculate_bollinger_bands(
        self, df: pd.DataFrame, window: int = 20
    ) -> Dict[str, Any]:
        """Calculate Bollinger Bands, returning both current values and full series."""
        close = df["Close"]

        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        current_close = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = rolling_mean.iloc[-1]

        if current_close > current_upper:
            position = "above_upper"
        elif current_close < current_lower:
            position = "below_lower"
        else:
            position = "within_bands"

        return {
            "current": {
                "upper_band": current_upper,
                "middle_band": current_middle,
                "lower_band": current_lower,
                "position": position,
                "width": (current_upper - current_lower) / current_middle,
            },
            "series": {
                "upper": upper_band,
                "middle": rolling_mean,
                "lower": lower_band,
            },
        }

    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility metrics. No time series — aggregate only."""
        returns = df["Close"].pct_change().dropna()
        volatility_daily = returns.std()
        volatility_annualized = volatility_daily * np.sqrt(252)

        return {
            "daily_volatility": volatility_daily,
            "annualized_volatility": volatility_annualized,
            "max_drawdown": (df["Close"] / df["Close"].cummax() - 1.0).min(),
        }

    # ── Financial metrics (unchanged) ───────────────────────────────────

    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert dictionary data back to DataFrame.

        Financial statement dicts (income_stmt, balance_sheet) from yfinance
        have string row labels as the index and dates as columns — do NOT
        coerce the index to DatetimeIndex.
        """
        if not data:
            return pd.DataFrame()
        try:
            return pd.DataFrame(
                data=data["data"],
                columns=data["columns"],
                index=data["index"],
            )
        except Exception as e:
            print(f"Error converting dictionary to DataFrame: {e}")
            return pd.DataFrame()

    def _find_row_by_labels(
        self, df: pd.DataFrame, labels: List[str]
    ) -> Optional[pd.Series]:
        """Find a row in DataFrame by trying multiple possible labels."""
        for label in labels:
            if label in df.index:
                return df.loc[label]
        return None

    def _calculate_financial_metrics(
        self, financials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate key financial metrics from financial statements"""
        metrics = {}

        revenue_labels = [
            "Total Revenue", "Total Revenues", "Revenue", "Revenues",
            "Net Revenue", "Operating Revenue",
        ]
        net_income_labels = [
            "Net Income", "Net Income Common Stockholders",
            "Net Income From Continuing Operations",
            "Net Income Applicable To Common Shares",
        ]
        total_assets_labels = ["Total Assets"]
        total_liabilities_labels = [
            "Total Liabilities Net Minority Interest", "Total Liabilities",
            "Total Debt", "Total Non Current Liabilities Net Minority Interest",
        ]

        try:
            income_stmt = self._dict_to_dataframe(financials.get("income_stmt", {}))
            if not income_stmt.empty:
                total_revenue = self._find_row_by_labels(income_stmt, revenue_labels)
                net_income = self._find_row_by_labels(income_stmt, net_income_labels)

                if total_revenue is not None and len(total_revenue) >= 2:
                    if total_revenue.iloc[1] != 0:
                        metrics["revenue_growth_yoy"] = (
                            (total_revenue.iloc[0] / total_revenue.iloc[1]) - 1
                        ) * 100

                if net_income is not None and len(net_income) >= 2:
                    if net_income.iloc[1] != 0:
                        metrics["net_income_growth_yoy"] = (
                            (net_income.iloc[0] / net_income.iloc[1]) - 1
                        ) * 100
        except Exception as e:
            print(f"Error calculating income statement metrics: {e}")

        try:
            balance_sheet = self._dict_to_dataframe(financials.get("balance_sheet", {}))
            if not balance_sheet.empty:
                total_assets = self._find_row_by_labels(balance_sheet, total_assets_labels)
                total_liabilities = self._find_row_by_labels(
                    balance_sheet, total_liabilities_labels
                )

                if total_assets is not None and total_liabilities is not None:
                    if total_assets.iloc[0] != 0:
                        metrics["debt_to_assets"] = (
                            total_liabilities.iloc[0] / total_assets.iloc[0]
                        ) * 100
        except Exception as e:
            print(f"Error calculating balance sheet metrics: {e}")

        return metrics
