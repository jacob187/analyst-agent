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

        result = {
            "moving_averages": raw["moving_averages"]["current"],
            "rsi": raw["rsi"]["current"],
            "macd": raw["macd"]["current"],
            "bollinger_bands": raw["bollinger_bands"]["current"],
            "volatility": raw["volatility"],
        }

        # Include advanced scalar indicators when available
        for key in ("adx", "atr", "stochastic", "volume_profile", "fibonacci"):
            if key in raw:
                result[key] = raw[key]

        return result

    def calculate_chart_indicators(
        self, df: Optional[pd.DataFrame], intraday: bool = False
    ) -> Dict[str, Any]:
        """Calculate indicators and return full time series for chart rendering.

        Args:
            df: OHLCV DataFrame.
            intraday: If True, use unix timestamps instead of date strings.

        Returns dict with keys like 'ma20', 'rsi', 'macd', 'bollinger' —
        each containing a list of {time, value} dicts ready for Lightweight Charts.
        """
        if df is None or df.empty:
            return {}

        self._intraday = intraday
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
        raw = {
            "moving_averages": self._calculate_moving_averages(df),
            "rsi": self._calculate_rsi(df),
            "macd": self._calculate_macd(df),
            "bollinger_bands": self._calculate_bollinger_bands(df),
            "volatility": self._calculate_volatility(df),
        }

        # Advanced indicators — each wrapped individually so one failure
        # doesn't break the rest.  These return scalar dicts only (no series).
        for key, calc_fn in [
            ("adx", self._calculate_adx),
            ("atr", self._calculate_atr),
            ("stochastic", self._calculate_stochastic),
            ("volume_profile", self._calculate_volume_profile),
            ("fibonacci", self._calculate_fibonacci_levels),
        ]:
            try:
                raw[key] = calc_fn(df)
            except Exception as e:
                print(f"Warning: {key} calculation failed for {self.ticker}: {e}")

        return raw

    # ── Chart format helpers ────────────────────────────────────────────

    def _format_time_column(self, index: pd.DatetimeIndex):
        """Format timestamps for Lightweight Charts.

        Daily data → "YYYY-MM-DD" strings.
        Intraday data → Unix timestamps (integers).
        """
        if getattr(self, "_intraday", False):
            return (index.astype("int64") // 10**9).tolist()
        return index.strftime("%Y-%m-%d").tolist()

    def _series_to_chart_format(
        self, series: pd.Series, index: pd.DatetimeIndex
    ) -> List[Dict[str, Any]]:
        """Convert a pandas Series to [{time, value}].

        Drops NaN and infinity entries (from rolling warmup periods).
        """
        times = self._format_time_column(index)
        tmp = pd.DataFrame({
            "time": times,
            "value": series.round(4),
        })
        tmp = tmp.dropna(subset=["value"])
        tmp = tmp[np.isfinite(tmp["value"])]
        return tmp.to_dict("records")

    def _macd_to_chart_format(
        self,
        macd_line: pd.Series,
        signal_line: pd.Series,
        histogram: pd.Series,
        index: pd.DatetimeIndex,
    ) -> List[Dict[str, Any]]:
        """Convert MACD series to [{time, macd, signal, histogram}]."""
        times = self._format_time_column(index)
        tmp = pd.DataFrame({
            "time": times,
            "macd": macd_line.round(4),
            "signal": signal_line.round(4),
            "histogram": histogram.round(4),
        })
        return tmp.dropna().to_dict("records")

    def _bollinger_to_chart_format(
        self,
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series,
        index: pd.DatetimeIndex,
    ) -> List[Dict[str, Any]]:
        """Convert Bollinger Band series to [{time, upper, middle, lower}]."""
        times = self._format_time_column(index)
        tmp = pd.DataFrame({
            "time": times,
            "upper": upper.round(4),
            "middle": middle.round(4),
            "lower": lower.round(4),
        })
        return tmp.dropna().to_dict("records")
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

    # ── Advanced indicators (scalar-only, no time series) ───────────────

    def _calculate_adx(self, df: pd.DataFrame, periods: int = 14) -> Dict[str, Any]:
        """Average Directional Index — measures trend strength regardless of direction.

        ADX is derived from two directional indicators (+DI and -DI) which track
        upward vs downward price movement.  The ADX itself is a smoothed average
        of the absolute difference between them, normalized by their sum.

        Interpretation:
            ADX < 20  -> weak / no trend (range-bound)
            20-40     -> developing trend
            40-60     -> strong trend
            > 60      -> very strong trend
        """
        if len(df) < periods * 2:
            return {}

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range components
        plus_dm = high.diff()
        minus_dm = low.diff().multiply(-1)

        # Keep only the larger directional move; zero out the other
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed averages (Wilder smoothing via EWM)
        atr_smooth = tr.ewm(alpha=1 / periods, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / periods, adjust=False).mean() / atr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / periods, adjust=False).mean() / atr_smooth)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
        adx = dx.ewm(alpha=1 / periods, adjust=False).mean()

        adx_val = float(adx.iloc[-1])

        if adx_val < 20:
            strength = "weak"
            signal = "Range-bound — mean-reversion strategies favored"
        elif adx_val < 40:
            strength = "developing"
            signal = "Trend developing — watch for breakout confirmation"
        elif adx_val < 60:
            strength = "strong"
            signal = "Strong trend — trend-following strategies favored"
        else:
            strength = "very_strong"
            signal = "Very strong trend — stay with the trend, avoid counter-trend"

        return {
            "adx": round(adx_val, 2),
            "plus_di": round(float(plus_di.iloc[-1]), 2),
            "minus_di": round(float(minus_di.iloc[-1]), 2),
            "trend_strength": strength,
            "signal": signal,
        }

    def _calculate_atr(self, df: pd.DataFrame, periods: int = 14) -> Dict[str, Any]:
        """Average True Range — measures volatility in price terms.

        ATR is not directional; it tells you *how much* price moves on average.
        Useful for:
        - Setting stop-loss distances (e.g. 2x ATR below entry)
        - Position sizing (higher ATR -> smaller position to cap risk)
        - Comparing volatility regimes over time
        """
        if len(df) < periods + 1:
            return {}

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / periods, adjust=False).mean()
        atr_val = float(atr.iloc[-1])
        current_price = float(close.iloc[-1])
        atr_pct = (atr_val / current_price) * 100 if current_price else 0

        # Suggested stop-loss: 2x ATR below current price
        stop_loss = current_price - (2 * atr_val)

        # Volatility regime from ATR percentile over its own history
        atr_rank = float((atr < atr_val).sum() / len(atr) * 100)
        if atr_rank < 25:
            regime = "low"
        elif atr_rank < 75:
            regime = "normal"
        else:
            regime = "high"

        return {
            "atr": round(atr_val, 4),
            "atr_percent": round(atr_pct, 2),
            "suggested_stop_loss": round(stop_loss, 2),
            "volatility_regime": regime,
        }

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, Any]:
        """Stochastic Oscillator — compares closing price to its range.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA(%K, d_period)

        The stochastic oscillates 0-100:
            < 20  -> oversold territory
            > 80  -> overbought territory
        Crossovers between %K and %D generate trading signals.
        """
        if len(df) < k_period + d_period:
            return {}

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        denom = highest_high - lowest_low
        denom = denom.replace(0, 1)  # avoid division by zero

        k_percent = 100 * (close - lowest_low) / denom
        d_percent = k_percent.rolling(window=d_period).mean()

        k_val = float(k_percent.iloc[-1])
        d_val = float(d_percent.iloc[-1])

        if k_val < 20:
            signal = "oversold"
        elif k_val > 80:
            signal = "overbought"
        else:
            signal = "neutral"

        # Crossover detection (last bar)
        prev_k = float(k_percent.iloc[-2])
        prev_d = float(d_percent.iloc[-2])
        if prev_k <= prev_d and k_val > d_val:
            crossover = "bullish"
        elif prev_k >= prev_d and k_val < d_val:
            crossover = "bearish"
        else:
            crossover = "none"

        return {
            "k_percent": round(k_val, 2),
            "d_percent": round(d_val, 2),
            "signal": signal,
            "crossover": crossover,
        }

    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> Dict[str, Any]:
        """Volume Profile — distribution of volume across price levels.

        Bins the price range into `bins` buckets and sums volume in each.
        Key outputs:
        - POC (Point of Control): price level with the most volume — acts as
          a magnet for price and a strong support/resistance zone.
        - Value Area: range containing ~70% of total volume.
        """
        if "Volume" not in df.columns or len(df) < 10:
            return {}

        price_min = float(df["Low"].min())
        price_max = float(df["High"].max())
        if price_min == price_max:
            return {}

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        vol_profile = np.zeros(bins)

        for price, vol in zip(typical_price.values, df["Volume"].values):
            idx = min(int((price - price_min) / (price_max - price_min) * bins), bins - 1)
            vol_profile[idx] += vol

        # POC — bin with highest volume
        poc_idx = int(np.argmax(vol_profile))
        poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

        # Value Area — expand outward from POC until 70% of volume is captured
        total_vol = vol_profile.sum()
        if total_vol == 0:
            return {}

        captured = vol_profile[poc_idx]
        lo, hi = poc_idx, poc_idx
        while captured / total_vol < 0.70 and (lo > 0 or hi < bins - 1):
            expand_lo = vol_profile[lo - 1] if lo > 0 else 0
            expand_hi = vol_profile[hi + 1] if hi < bins - 1 else 0
            if expand_lo >= expand_hi and lo > 0:
                lo -= 1
                captured += expand_lo
            elif hi < bins - 1:
                hi += 1
                captured += expand_hi
            else:
                lo -= 1
                captured += expand_lo

        va_high = float(bin_edges[hi + 1])
        va_low = float(bin_edges[lo])
        current_price = float(df["Close"].iloc[-1])

        if current_price > va_high:
            position = "above_value_area"
        elif current_price < va_low:
            position = "below_value_area"
        else:
            position = "inside_value_area"

        return {
            "poc": round(poc, 2),
            "value_area_high": round(va_high, 2),
            "value_area_low": round(va_low, 2),
            "position": position,
        }

    def _calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """Fibonacci Retracement levels from recent swing high/low.

        Fibonacci ratios (23.6%, 38.2%, 50%, 61.8%, 78.6%) are projected
        between the highest high and lowest low of the lookback window.
        Traders watch these levels for potential reversal or continuation zones.
        """
        if len(df) < min(lookback, 10):
            return {}

        window = df.tail(min(lookback, len(df)))
        swing_high = float(window["High"].max())
        swing_low = float(window["Low"].min())
        diff = swing_high - swing_low
        if diff == 0:
            return {}

        fib_ratios = {"0.236": 0.236, "0.382": 0.382, "0.500": 0.5, "0.618": 0.618, "0.786": 0.786}
        levels = {}
        for label, ratio in fib_ratios.items():
            levels[label] = round(swing_high - diff * ratio, 2)

        current_price = float(df["Close"].iloc[-1])

        # Find the closest Fibonacci level
        closest_label = min(levels, key=lambda k: abs(levels[k] - current_price))
        closest_price = levels[closest_label]
        distance = round(abs(current_price - closest_price) / current_price * 100, 2)

        return {
            "levels": levels,
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "current_price": round(current_price, 2),
            "closest_level": closest_label,
            "closest_price": closest_price,
            "distance_to_closest": distance,
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
