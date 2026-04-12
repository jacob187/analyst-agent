"""Chart pattern detection engine.

Identifies classical technical patterns (Head & Shoulders, Double Top/Bottom,
MA crossovers, RSI divergences) from OHLCV DataFrames. Each detector returns
a list of pattern dicts with a common schema so callers can iterate uniformly.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class PatternRecognitionEngine:
    """Detect chart patterns from OHLCV price data.

    All detectors are independent — a failure in one never blocks the others.
    Each pattern dict has at minimum: {type, direction, confidence, status}.
    """

    def detect_all_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run every detector and merge results into a single list.

        Contradictory patterns (e.g. H&S + inverse H&S, or double top +
        double bottom) are resolved by keeping only the higher-confidence one.
        """
        if df is None or df.empty:
            return []

        patterns: List[Dict[str, Any]] = []
        detectors = [
            self._detect_head_and_shoulders,
            self._detect_double_top_bottom,
            self._detect_ma_crossovers,
            self._detect_divergences,
        ]
        for detector in detectors:
            try:
                patterns.extend(detector(df))
            except Exception as e:
                print(f"Pattern detector {detector.__name__} failed: {e}")

        patterns = self._resolve_contradictions(patterns)
        return patterns

    def _resolve_contradictions(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove contradictory patterns, keeping only the higher-confidence one.

        Two patterns contradict when they imply opposite directions on the same
        price structure — e.g. Head & Shoulders (bearish) vs Inverse H&S (bullish),
        or Double Top (bearish) vs Double Bottom (bullish).
        """
        contradiction_groups = [
            {"head_and_shoulders", "inverse_head_and_shoulders"},
            {"double_top", "double_bottom"},
        ]

        to_remove: set[str] = set()
        for group in contradiction_groups:
            group_patterns = [p for p in patterns if p["type"] in group]
            if len(group_patterns) >= 2:
                # Keep only the highest-confidence pattern in this group
                best = max(group_patterns, key=lambda p: p["confidence"])
                for p in group_patterns:
                    if p is not best:
                        to_remove.add(id(p))

        return [p for p in patterns if id(p) not in to_remove]

    # ── Individual detectors ──────────────────────────────────────────────

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Look at last 50 bars for three peaks where the middle is highest.

        A Head & Shoulders top requires:
        1. Three peaks (local maxima) in the recent window.
        2. The middle peak (head) is higher than both shoulders.
        3. Shoulders are roughly equal (within 3% of each other).

        An inverse H&S mirrors this logic on troughs.
        """
        if len(df) < 50:
            return []

        window = df.tail(50)
        highs = window["High"].values
        lows = window["Low"].values
        patterns = []

        # Find local maxima / minima with a simple rolling comparison (order=3)
        max_indices = self._find_local_extrema(highs, order=3, mode="max")
        min_indices = self._find_local_extrema(lows, order=3, mode="min")

        # H&S Top — need at least 3 peaks
        if len(max_indices) >= 3:
            for i in range(len(max_indices) - 2):
                left, mid, right = max_indices[i], max_indices[i + 1], max_indices[i + 2]
                left_val, mid_val, right_val = highs[left], highs[mid], highs[right]

                if mid_val > left_val and mid_val > right_val:
                    shoulder_diff = abs(left_val - right_val) / max(left_val, right_val)
                    if shoulder_diff < 0.03:
                        neckline = min(lows[left:right + 1])
                        geometry = self._hs_geometry_score(
                            left_val, mid_val, right_val, shoulder_diff
                        )
                        confidence = self._calculate_pattern_confidence(
                            df, "head_and_shoulders",
                            len(df) - 50 + left, len(df) - 50 + right,
                            geometry_score=geometry,
                        )
                        patterns.append({
                            "type": "head_and_shoulders",
                            "direction": "bearish",
                            "confidence": confidence,
                            "status": "forming" if right >= len(highs) - 5 else "completed",
                            "head_price": float(mid_val),
                            "neckline": float(neckline),
                            "target": float(neckline - (mid_val - neckline)),
                        })
                        break  # report best match only

        # Inverse H&S — need at least 3 troughs
        if len(min_indices) >= 3:
            for i in range(len(min_indices) - 2):
                left, mid, right = min_indices[i], min_indices[i + 1], min_indices[i + 2]
                left_val, mid_val, right_val = lows[left], lows[mid], lows[right]

                if mid_val < left_val and mid_val < right_val:
                    shoulder_diff = abs(left_val - right_val) / max(left_val, right_val)
                    if shoulder_diff < 0.03:
                        neckline = max(highs[left:right + 1])
                        geometry = self._hs_geometry_score(
                            left_val, mid_val, right_val, shoulder_diff
                        )
                        confidence = self._calculate_pattern_confidence(
                            df, "inverse_head_and_shoulders",
                            len(df) - 50 + left, len(df) - 50 + right,
                            geometry_score=geometry,
                        )
                        patterns.append({
                            "type": "inverse_head_and_shoulders",
                            "direction": "bullish",
                            "confidence": confidence,
                            "status": "forming" if right >= len(lows) - 5 else "completed",
                            "head_price": float(mid_val),
                            "neckline": float(neckline),
                            "target": float(neckline + (neckline - mid_val)),
                        })
                        break

        return patterns

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Look at last 40 bars for two peaks/troughs within 2% of each other.

        Double Top (bearish): two highs at roughly the same level with a valley.
        Double Bottom (bullish): two lows at roughly the same level with a peak.
        """
        if len(df) < 40:
            return []

        window = df.tail(40)
        highs = window["High"].values
        lows = window["Low"].values
        patterns = []

        max_indices = self._find_local_extrema(highs, order=3, mode="max")
        min_indices = self._find_local_extrema(lows, order=3, mode="min")

        # Double Top
        if len(max_indices) >= 2:
            for i in range(len(max_indices) - 1):
                a, b = max_indices[i], max_indices[i + 1]
                a_val, b_val = highs[a], highs[b]
                diff_pct = abs(a_val - b_val) / max(a_val, b_val)
                if diff_pct < 0.02 and (b - a) >= 5:
                    valley = min(lows[a:b + 1])
                    peak_avg = (a_val + b_val) / 2
                    confidence = self._calculate_pattern_confidence(
                        df, "double_top", len(df) - 40 + a, len(df) - 40 + b
                    )
                    patterns.append({
                        "type": "double_top",
                        "direction": "bearish",
                        "confidence": confidence,
                        "status": "forming" if b >= len(highs) - 5 else "completed",
                        "peak_price": float(peak_avg),
                        "valley": float(valley),
                        "target": float(valley - (peak_avg - valley)),
                    })
                    break

        # Double Bottom
        if len(min_indices) >= 2:
            for i in range(len(min_indices) - 1):
                a, b = min_indices[i], min_indices[i + 1]
                a_val, b_val = lows[a], lows[b]
                diff_pct = abs(a_val - b_val) / max(a_val, b_val)
                if diff_pct < 0.02 and (b - a) >= 5:
                    peak = max(highs[a:b + 1])
                    trough_avg = (a_val + b_val) / 2
                    confidence = self._calculate_pattern_confidence(
                        df, "double_bottom", len(df) - 40 + a, len(df) - 40 + b
                    )
                    patterns.append({
                        "type": "double_bottom",
                        "direction": "bullish",
                        "confidence": confidence,
                        "status": "forming" if b >= len(lows) - 5 else "completed",
                        "trough_price": float(trough_avg),
                        "peak": float(peak),
                        "target": float(peak + (peak - trough_avg)),
                    })
                    break

        return patterns

    def _detect_ma_crossovers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Golden Cross (MA50 crosses above MA200) and Death Cross in last 5 days.

        Requires at least 200 rows of data to compute both moving averages.
        Only looks at the most recent 5 bars for the actual crossover event.
        """
        if len(df) < 200:
            return []

        close = df["Close"]
        ma50 = close.rolling(window=50).mean()
        ma200 = close.rolling(window=200).mean()
        patterns = []

        # Check last 5 days for a crossover
        for i in range(-5, 0):
            prev_diff = ma50.iloc[i - 1] - ma200.iloc[i - 1]
            curr_diff = ma50.iloc[i] - ma200.iloc[i]

            if prev_diff <= 0 and curr_diff > 0:
                patterns.append({
                    "type": "golden_cross",
                    "direction": "bullish",
                    "confidence": 0.75,
                    "status": "confirmed",
                    "ma50": float(ma50.iloc[i]),
                    "ma200": float(ma200.iloc[i]),
                    "cross_date": str(df.index[i])[:10],
                })
            elif prev_diff >= 0 and curr_diff < 0:
                patterns.append({
                    "type": "death_cross",
                    "direction": "bearish",
                    "confidence": 0.75,
                    "status": "confirmed",
                    "ma50": float(ma50.iloc[i]),
                    "ma200": float(ma200.iloc[i]),
                    "cross_date": str(df.index[i])[:10],
                })

        return patterns

    def _detect_divergences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect RSI divergence over recent bars.

        Bullish divergence: price makes a lower low while RSI makes a higher low.
        Bearish divergence: price makes a higher high while RSI makes a lower high.

        Uses a 20-bar lookback window split into two halves to compare.
        """
        if len(df) < 30:
            return []

        close = df["Close"]
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(com=13, adjust=False).mean()
        roll_down = down.ewm(com=13, adjust=False).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        patterns = []
        window = 20
        recent_close = close.iloc[-window:]
        recent_rsi = rsi.iloc[-window:]

        # Split into two halves
        half = window // 2
        first_close = recent_close.iloc[:half]
        second_close = recent_close.iloc[half:]
        first_rsi = recent_rsi.iloc[:half]
        second_rsi = recent_rsi.iloc[half:]

        # Bullish divergence: lower low in price, higher low in RSI
        price_low1 = first_close.min()
        price_low2 = second_close.min()
        rsi_low1 = first_rsi.min()
        rsi_low2 = second_rsi.min()

        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            strength = (rsi_low2 - rsi_low1) / 100
            patterns.append({
                "type": "bullish_divergence",
                "direction": "bullish",
                "confidence": min(0.5 + strength * 2, 0.9),
                "status": "active",
                "price_low": float(price_low2),
                "rsi_low": float(rsi_low2),
            })

        # Bearish divergence: higher high in price, lower high in RSI
        price_high1 = first_close.max()
        price_high2 = second_close.max()
        rsi_high1 = first_rsi.max()
        rsi_high2 = second_rsi.max()

        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
            strength = (rsi_high1 - rsi_high2) / 100
            patterns.append({
                "type": "bearish_divergence",
                "direction": "bearish",
                "confidence": min(0.5 + strength * 2, 0.9),
                "status": "active",
                "price_high": float(price_high2),
                "rsi_high": float(rsi_high2),
            })

        return patterns

    # ── Helpers ────────────────────────────────────────────────────────────

    def _hs_geometry_score(
        self,
        left_val: float,
        mid_val: float,
        right_val: float,
        shoulder_diff: float,
    ) -> float:
        """Score how well three peaks/troughs match ideal H&S geometry.

        Two factors:
        1. Head prominence — how far the head extends beyond the shoulders.
           A head that's barely above the shoulders (e.g. 0.5%) is weak;
           one that's 3%+ beyond is textbook.
        2. Shoulder symmetry — already measured as shoulder_diff (0 = perfect).

        Returns 0.0–1.0 where 1.0 is a textbook pattern.
        """
        shoulder_avg = (left_val + right_val) / 2
        # Head prominence: % the head extends beyond shoulder average
        prominence = abs(mid_val - shoulder_avg) / shoulder_avg if shoulder_avg else 0

        # Map prominence to 0–1: <0.5% → 0.1, 1.5% → 0.5, 3%+ → 1.0
        if prominence < 0.005:
            prominence_score = 0.1
        elif prominence < 0.015:
            prominence_score = 0.1 + (prominence - 0.005) * 40  # 0.1 → 0.5
        elif prominence < 0.03:
            prominence_score = 0.5 + (prominence - 0.015) * 33.3  # 0.5 → 1.0
        else:
            prominence_score = 1.0

        # Symmetry: shoulder_diff of 0 → 1.0, 0.03 → 0.0
        symmetry_score = max(0, 1.0 - shoulder_diff / 0.03)

        return 0.6 * prominence_score + 0.4 * symmetry_score

    def _find_local_extrema(
        self, values: np.ndarray, order: int = 3, mode: str = "max"
    ) -> List[int]:
        """Find local extrema using a simple rolling comparison.

        A point is a local max if it's >= all neighbours within `order` bars
        on each side. Same logic inverted for min. No scipy dependency.
        """
        indices = []
        for i in range(order, len(values) - order):
            window = values[i - order: i + order + 1]
            if mode == "max" and values[i] == window.max():
                indices.append(i)
            elif mode == "min" and values[i] == window.min():
                indices.append(i)
        return indices

    def _calculate_pattern_confidence(
        self, df: pd.DataFrame, pattern_type: str, start_idx: int, end_idx: int,
        geometry_score: float = 0.5,
    ) -> float:
        """Score confidence from 0-1 using volume confirmation and pattern geometry.

        Two components (equally weighted):
        1. Volume: higher avg volume during the pattern window (vs 50-bar baseline)
           signals institutional participation.
        2. Geometry: caller passes a 0-1 score reflecting how well the price
           structure matches the ideal pattern shape (shoulder symmetry, head
           prominence, etc.).

        Args:
            geometry_score: How well the price structure matches the ideal
                pattern. 1.0 = textbook shape, 0.0 = barely qualifies.
                Defaults to 0.5 when not provided (backwards compat).
        """
        start_idx = max(0, start_idx)
        end_idx = min(len(df) - 1, end_idx)

        # Volume component
        vol_score = 0.5
        if "Volume" in df.columns:
            pattern_vol = df["Volume"].iloc[start_idx:end_idx + 1].mean()
            baseline_start = max(0, start_idx - 50)
            baseline_vol = df["Volume"].iloc[baseline_start:start_idx].mean()
            if baseline_vol > 0:
                vol_ratio = pattern_vol / baseline_vol
                # Map ratio to 0.2–0.9: ratio 0.5 → 0.35, 1.0 → 0.5, 2.0 → 0.8
                vol_score = min(0.2 + vol_ratio * 0.3, 0.9)

        # Blend: 50% geometry, 50% volume
        confidence = 0.5 * geometry_score + 0.5 * vol_score
        # Clamp to [0.2, 0.9] — never fully certain or fully dismissive
        return round(max(0.2, min(confidence, 0.9)), 2)
