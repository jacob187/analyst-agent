"""Support and resistance level detection.

Uses four independent methods — extrema, volume profile, round numbers, and
Fibonacci retracement — then merges nearby levels and ranks them by a
composite score (touches, volume, recency, source count).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class SupportResistanceDetector:
    """Detect, merge, and rank support/resistance levels from OHLCV data."""

    def detect_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all four detection methods, merge, rank, and split into S/R.

        Returns:
            {current_price, support_levels: [...], resistance_levels: [...]}
            Each level dict: {price, score, sources, touches}
        """
        if df is None or df.empty:
            return {"current_price": 0, "support_levels": [], "resistance_levels": []}

        current_price = float(df["Close"].iloc[-1])

        # Collect raw levels from each method — each returns List[Dict]
        raw_levels: List[Dict[str, Any]] = []
        detectors = [
            ("extrema", self._find_extrema_levels),
            ("volume", self._find_volume_levels),
            ("round_number", self._find_round_number_levels),
            ("fibonacci", self._find_fibonacci_levels),
        ]
        for source_name, fn in detectors:
            try:
                levels = fn(df)
                for lvl in levels:
                    lvl["source"] = source_name
                raw_levels.extend(levels)
            except Exception as e:
                print(f"S/R detector {source_name} failed: {e}")

        merged = self._merge_and_rank_levels(raw_levels, df)

        support = [l for l in merged if l["price"] < current_price]
        resistance = [l for l in merged if l["price"] >= current_price]

        # Sort: support descending (closest first), resistance ascending
        support.sort(key=lambda x: x["price"], reverse=True)
        resistance.sort(key=lambda x: x["price"])

        return {
            "current_price": current_price,
            "support_levels": support[:5],
            "resistance_levels": resistance[:5],
        }

    # ── Detection methods ─────────────────────────────────────────────────

    def _find_extrema_levels(self, df: pd.DataFrame, order: int = 5) -> List[Dict[str, Any]]:
        """Find local maxima/minima using rolling comparison (no scipy).

        A point is an extremum if it's the max/min within `order` bars on
        each side. Returns the price at each extremum.
        """
        levels = []
        highs = df["High"].values
        lows = df["Low"].values

        for i in range(order, len(highs) - order):
            window_high = highs[i - order: i + order + 1]
            if highs[i] == window_high.max():
                levels.append({"price": float(highs[i]), "type": "resistance"})

            window_low = lows[i - order: i + order + 1]
            if lows[i] == window_low.min():
                levels.append({"price": float(lows[i]), "type": "support"})

        return levels

    def _find_volume_levels(self, df: pd.DataFrame, bins: int = 50) -> List[Dict[str, Any]]:
        """Bin price range by volume to find high-volume zones.

        These zones act as support/resistance because institutional orders
        cluster at prices where lots of volume has traded.
        """
        if "Volume" not in df.columns or len(df) < 10:
            return []

        price_min = df["Low"].min()
        price_max = df["High"].max()
        if price_min == price_max:
            return []

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        # Assign each bar's volume to the bin matching its typical price
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vol_profile = np.zeros(bins)

        for price, vol in zip(typical.values, df["Volume"].values):
            idx = min(int((price - price_min) / (price_max - price_min) * bins), bins - 1)
            vol_profile[idx] += vol

        # Top 5 volume bins
        threshold = np.percentile(vol_profile, 80)
        levels = []
        for i in range(bins):
            if vol_profile[i] >= threshold:
                mid_price = (bin_edges[i] + bin_edges[i + 1]) / 2
                levels.append({
                    "price": float(round(mid_price, 2)),
                    "type": "volume_node",
                    "volume": float(vol_profile[i]),
                })
        return levels

    def _find_round_number_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate psychological round-number levels near the current price.

        The step size scales with price magnitude — $1 steps for low-priced
        stocks, $50 or $100 steps for high-priced ones.
        """
        current = float(df["Close"].iloc[-1])
        if current <= 0:
            return []

        # Choose step based on price magnitude
        if current < 10:
            step = 1
        elif current < 50:
            step = 5
        elif current < 200:
            step = 10
        elif current < 500:
            step = 25
        elif current < 1000:
            step = 50
        else:
            step = 100

        # Generate levels within ±20% of current price
        low = current * 0.80
        high = current * 1.20
        start = int(low / step) * step

        levels = []
        price = start
        while price <= high:
            if price > 0:
                levels.append({"price": float(price), "type": "round_number"})
            price += step
        return levels

    def _find_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict[str, Any]]:
        """Calculate Fibonacci retracement levels from swing high/low.

        Uses the highest high and lowest low over the lookback window,
        then projects standard Fibonacci ratios.
        """
        if len(df) < lookback:
            lookback = len(df)

        window = df.tail(lookback)
        swing_high = float(window["High"].max())
        swing_low = float(window["Low"].min())
        diff = swing_high - swing_low
        if diff == 0:
            return []

        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = []
        for ratio in fib_ratios:
            price = swing_high - diff * ratio
            levels.append({
                "price": float(round(price, 2)),
                "type": "fibonacci",
                "ratio": ratio,
            })
        return levels

    # ── Merge and rank ────────────────────────────────────────────────────

    def _merge_and_rank_levels(
        self,
        levels: List[Dict[str, Any]],
        df: pd.DataFrame,
        tolerance: float = 0.02,
    ) -> List[Dict[str, Any]]:
        """Merge levels within `tolerance` (2%) of each other, then score.

        Scoring factors:
        - sources: number of distinct detection methods (max contribution)
        - touches: how many bars came within 0.5% of this level
        - volume: average volume near the level
        - recency: bonus if the level was touched in the last 20 bars
        """
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        merged: List[Dict[str, Any]] = []

        current_group = [sorted_levels[0]]
        for lvl in sorted_levels[1:]:
            group_avg = np.mean([l["price"] for l in current_group])
            if abs(lvl["price"] - group_avg) / max(group_avg, 1e-9) <= tolerance:
                current_group.append(lvl)
            else:
                merged.append(self._score_group(current_group, df))
                current_group = [lvl]
        merged.append(self._score_group(current_group, df))

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged

    def _score_group(
        self, group: List[Dict[str, Any]], df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Score a merged level group."""
        avg_price = float(np.mean([l["price"] for l in group]))
        sources = list({l.get("source", "unknown") for l in group})

        # Count touches — bars where close came within 0.5% of the level
        close = df["Close"].values
        touch_threshold = avg_price * 0.005
        touches = int(np.sum(np.abs(close - avg_price) <= touch_threshold))

        # Recency bonus — touched in last 20 bars?
        recent_close = close[-20:] if len(close) >= 20 else close
        recency = 1.0 if np.any(np.abs(recent_close - avg_price) <= touch_threshold) else 0.0

        # Composite score
        score = len(sources) * 2.0 + touches * 0.5 + recency * 1.5

        return {
            "price": round(avg_price, 2),
            "score": round(score, 2),
            "sources": sources,
            "touches": touches,
        }
