"""Tests for SupportResistanceDetector.

Uses synthetic DataFrames to test each detection method and the merge/rank
logic.  All tests are eval_unit (no API keys needed).
"""

import numpy as np
import pandas as pd
import pytest

from agents.technical_workflow.support_resistance import SupportResistanceDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector():
    return SupportResistanceDetector()


@pytest.fixture
def synthetic_df():
    """200-row OHLCV DataFrame with a range-bound price around $100."""
    np.random.seed(77)
    n = 200
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.random.normal(0, 3, n)
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# detect_levels (integration)
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestDetectLevels:
    def test_return_structure(self, detector, synthetic_df):
        result = detector.detect_levels(synthetic_df)
        assert "current_price" in result
        assert "support_levels" in result
        assert "resistance_levels" in result
        assert isinstance(result["support_levels"], list)
        assert isinstance(result["resistance_levels"], list)

    def test_support_below_price(self, detector, synthetic_df):
        result = detector.detect_levels(synthetic_df)
        price = result["current_price"]
        for s in result["support_levels"]:
            assert s["price"] < price

    def test_resistance_above_or_equal(self, detector, synthetic_df):
        result = detector.detect_levels(synthetic_df)
        price = result["current_price"]
        for r in result["resistance_levels"]:
            assert r["price"] >= price

    def test_empty_df(self, detector):
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = detector.detect_levels(empty)
        assert result == {"current_price": 0, "support_levels": [], "resistance_levels": []}

    def test_none_df(self, detector):
        result = detector.detect_levels(None)
        assert result["current_price"] == 0


# ---------------------------------------------------------------------------
# _find_extrema_levels
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestExtremaLevels:
    def test_returns_list(self, detector, synthetic_df):
        levels = detector._find_extrema_levels(synthetic_df)
        assert isinstance(levels, list)

    def test_level_has_price(self, detector, synthetic_df):
        levels = detector._find_extrema_levels(synthetic_df)
        for lvl in levels:
            assert "price" in lvl
            assert isinstance(lvl["price"], float)


# ---------------------------------------------------------------------------
# _find_volume_levels
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestVolumeLevels:
    def test_returns_list(self, detector, synthetic_df):
        levels = detector._find_volume_levels(synthetic_df)
        assert isinstance(levels, list)

    def test_no_volume_column(self, detector):
        df = pd.DataFrame(
            {"High": range(20), "Low": range(20), "Close": range(20)},
            index=pd.bdate_range("2025-01-01", periods=20),
        )
        assert detector._find_volume_levels(df) == []


# ---------------------------------------------------------------------------
# _find_round_number_levels
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestRoundNumberLevels:
    def test_low_price(self, detector):
        """Stock at $8 should get $1 steps."""
        df = pd.DataFrame(
            {"Close": [8.0]}, index=pd.bdate_range("2025-01-01", periods=1)
        )
        levels = detector._find_round_number_levels(df)
        prices = [l["price"] for l in levels]
        # All should be integers (step=1)
        assert all(p == int(p) for p in prices)

    def test_high_price(self, detector):
        """Stock at $1500 should get $100 steps."""
        df = pd.DataFrame(
            {"Close": [1500.0]}, index=pd.bdate_range("2025-01-01", periods=1)
        )
        levels = detector._find_round_number_levels(df)
        prices = [l["price"] for l in levels]
        assert all(p % 100 == 0 for p in prices)

    def test_mid_price(self, detector):
        """Stock at $75 should get $10 steps."""
        df = pd.DataFrame(
            {"Close": [75.0]}, index=pd.bdate_range("2025-01-01", periods=1)
        )
        levels = detector._find_round_number_levels(df)
        prices = [l["price"] for l in levels]
        assert all(p % 10 == 0 for p in prices)


# ---------------------------------------------------------------------------
# _find_fibonacci_levels
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestFibonacciLevels:
    def test_returns_5_levels(self, detector, synthetic_df):
        levels = detector._find_fibonacci_levels(synthetic_df)
        assert len(levels) == 5

    def test_levels_have_ratio(self, detector, synthetic_df):
        levels = detector._find_fibonacci_levels(synthetic_df)
        for lvl in levels:
            assert "ratio" in lvl


# ---------------------------------------------------------------------------
# _merge_and_rank_levels
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestMergeAndRank:
    def test_merges_nearby_levels(self, detector, synthetic_df):
        """Two levels within 2% should merge into one."""
        levels = [
            {"price": 100.0, "source": "extrema"},
            {"price": 101.5, "source": "volume"},  # 1.5% away from 100
        ]
        merged = detector._merge_and_rank_levels(levels, synthetic_df, tolerance=0.02)
        assert len(merged) == 1
        assert len(merged[0]["sources"]) == 2

    def test_keeps_distant_levels(self, detector, synthetic_df):
        """Two levels >2% apart should remain separate."""
        levels = [
            {"price": 100.0, "source": "extrema"},
            {"price": 110.0, "source": "volume"},
        ]
        merged = detector._merge_and_rank_levels(levels, synthetic_df, tolerance=0.02)
        assert len(merged) == 2

    def test_empty_input(self, detector, synthetic_df):
        assert detector._merge_and_rank_levels([], synthetic_df) == []

    def test_multi_source_has_more_sources(self, detector, synthetic_df):
        """A merged group from two detection methods should list both sources."""
        double = [
            {"price": 100.0, "source": "extrema"},
            {"price": 100.5, "source": "volume"},
        ]
        merged = detector._merge_and_rank_levels(double, synthetic_df)
        assert len(merged) == 1
        assert set(merged[0]["sources"]) == {"extrema", "volume"}
