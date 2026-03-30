"""Tests for the PatternRecognitionEngine.

Uses synthetic DataFrames to test each detector in isolation plus the
aggregate detect_all_patterns method.  All tests are eval_unit (no API keys).
"""

import numpy as np
import pandas as pd
import pytest

from agents.technical_workflow.pattern_recognition import PatternRecognitionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_PATTERN_KEYS = {"type", "direction", "confidence", "status"}


@pytest.fixture
def engine():
    return PatternRecognitionEngine()


@pytest.fixture
def synthetic_df():
    """200-row OHLCV DataFrame with random walk."""
    np.random.seed(99)
    n = 200
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, n))
    high = close + np.abs(np.random.normal(0.5, 0.3, n))
    low = close - np.abs(np.random.normal(0.5, 0.3, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def crossover_df():
    """DataFrame engineered to have a Golden Cross in the last 5 days.

    We construct the price so that MA50 is just below MA200 at bar -6, then
    rises above MA200 at bar -5 through -1.  To achieve this we need:
    - A first chunk where prices are high (pushes MA200 up)
    - A dip that drags MA50 below MA200
    - A recovery in the last bars that pushes MA50 above MA200
    """
    n = 250
    dates = pd.bdate_range("2024-01-01", periods=n)
    # High base for first 100 bars to build MA200
    chunk1 = np.full(100, 100.0)
    # Drop to 60 for next 100 bars — drags MA50 well below MA200
    chunk2 = np.full(100, 60.0)
    # Gradual recovery to 90 over 45 bars — brings MA50 close to MA200
    chunk3 = np.linspace(60, 90, 45)
    # Sharp spike in last 5 bars — pushes MA50 above MA200
    chunk4 = np.array([100, 110, 120, 130, 140])
    close = np.concatenate([chunk1, chunk2, chunk3, chunk4])
    high = close + 0.5
    low = close - 0.5
    volume = np.full(n, 5_000_000.0)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# detect_all_patterns
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestDetectAllPatterns:
    def test_returns_list(self, engine, synthetic_df):
        result = engine.detect_all_patterns(synthetic_df)
        assert isinstance(result, list)

    def test_pattern_dict_schema(self, engine, synthetic_df):
        patterns = engine.detect_all_patterns(synthetic_df)
        for p in patterns:
            assert REQUIRED_PATTERN_KEYS.issubset(p.keys()), f"Missing keys in {p}"

    def test_empty_df(self, engine):
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        assert engine.detect_all_patterns(empty) == []

    def test_none_df(self, engine):
        assert engine.detect_all_patterns(None) == []


# ---------------------------------------------------------------------------
# Individual detectors — insufficient data
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestInsufficientData:
    def test_head_and_shoulders_short(self, engine):
        short = pd.DataFrame(
            {"High": [1, 2, 3], "Low": [0.5, 1, 2], "Close": [0.8, 1.5, 2.5], "Volume": [100, 200, 300]},
            index=pd.bdate_range("2025-01-01", periods=3),
        )
        assert engine._detect_head_and_shoulders(short) == []

    def test_double_top_bottom_short(self, engine):
        short = pd.DataFrame(
            {"High": [1, 2], "Low": [0.5, 1], "Close": [0.8, 1.5], "Volume": [100, 200]},
            index=pd.bdate_range("2025-01-01", periods=2),
        )
        assert engine._detect_double_top_bottom(short) == []

    def test_ma_crossovers_short(self, engine):
        short = pd.DataFrame(
            {"Close": list(range(50))},
            index=pd.bdate_range("2025-01-01", periods=50),
        )
        assert engine._detect_ma_crossovers(short) == []

    def test_divergences_short(self, engine):
        short = pd.DataFrame(
            {"Close": [1, 2, 3]},
            index=pd.bdate_range("2025-01-01", periods=3),
        )
        assert engine._detect_divergences(short) == []


# ---------------------------------------------------------------------------
# MA crossover with engineered data
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
def test_golden_cross_detected(engine, crossover_df):
    """The crossover_df is designed so MA50 crosses above MA200 near the end."""
    patterns = engine._detect_ma_crossovers(crossover_df)
    # We expect at least one golden_cross (the ramp forces the crossover)
    golden = [p for p in patterns if p["type"] == "golden_cross"]
    assert len(golden) >= 1
    assert golden[0]["direction"] == "bullish"
    assert golden[0]["confidence"] == 0.75


# ---------------------------------------------------------------------------
# find_local_extrema helper
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
def test_find_local_extrema_max(engine):
    values = np.array([1, 2, 5, 2, 1, 3, 1])
    maxima = engine._find_local_extrema(values, order=1, mode="max")
    assert 2 in maxima  # index 2 is the peak (value 5)


@pytest.mark.eval_unit
def test_find_local_extrema_min(engine):
    values = np.array([5, 3, 1, 3, 5, 2, 5])
    minima = engine._find_local_extrema(values, order=1, mode="min")
    assert 2 in minima  # index 2 is the trough (value 1)
