"""Tests for the 5 new indicator methods on TechnicalIndicators.

Each test uses a synthetic OHLCV DataFrame so we can run without API keys
(marked eval_unit).  We verify return types, required keys, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from agents.technical_workflow.process_technical_indicators import TechnicalIndicators


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df():
    """200-row OHLCV DataFrame with a gentle uptrend and realistic volume.

    Close follows a random walk with upward drift so that all indicators
    have enough data to compute.  High/Low are offset from Close by a
    small random amount to simulate intraday range.
    """
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, n))
    high = close + np.abs(np.random.normal(0.5, 0.3, n))
    low = close - np.abs(np.random.normal(0.5, 0.3, n))
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def tech():
    return TechnicalIndicators("TEST")


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestADX:
    def test_return_keys(self, tech, synthetic_df):
        result = tech._calculate_adx(synthetic_df)
        assert isinstance(result, dict)
        for key in ("adx", "plus_di", "minus_di", "trend_strength", "signal"):
            assert key in result, f"Missing key: {key}"

    def test_adx_range(self, tech, synthetic_df):
        result = tech._calculate_adx(synthetic_df)
        assert 0 <= result["adx"] <= 100

    def test_trend_strength_label(self, tech, synthetic_df):
        result = tech._calculate_adx(synthetic_df)
        assert result["trend_strength"] in ("weak", "developing", "strong", "very_strong")

    def test_insufficient_data(self, tech):
        short_df = pd.DataFrame(
            {"High": [1, 2], "Low": [0.5, 1], "Close": [0.8, 1.5]},
            index=pd.bdate_range("2025-01-01", periods=2),
        )
        assert tech._calculate_adx(short_df) == {}

    def test_empty_df(self, tech):
        empty = pd.DataFrame(columns=["High", "Low", "Close"])
        assert tech._calculate_adx(empty) == {}


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestATR:
    def test_return_keys(self, tech, synthetic_df):
        result = tech._calculate_atr(synthetic_df)
        for key in ("atr", "atr_percent", "suggested_stop_loss", "volatility_regime"):
            assert key in result

    def test_atr_positive(self, tech, synthetic_df):
        result = tech._calculate_atr(synthetic_df)
        assert result["atr"] > 0

    def test_stop_loss_below_price(self, tech, synthetic_df):
        result = tech._calculate_atr(synthetic_df)
        current = float(synthetic_df["Close"].iloc[-1])
        assert result["suggested_stop_loss"] < current

    def test_volatility_regime_label(self, tech, synthetic_df):
        result = tech._calculate_atr(synthetic_df)
        assert result["volatility_regime"] in ("low", "normal", "high")

    def test_insufficient_data(self, tech):
        short = pd.DataFrame(
            {"High": [10], "Low": [9], "Close": [9.5]},
            index=pd.bdate_range("2025-01-01", periods=1),
        )
        assert tech._calculate_atr(short) == {}


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestStochastic:
    def test_return_keys(self, tech, synthetic_df):
        result = tech._calculate_stochastic(synthetic_df)
        for key in ("k_percent", "d_percent", "signal", "crossover"):
            assert key in result

    def test_k_d_range(self, tech, synthetic_df):
        result = tech._calculate_stochastic(synthetic_df)
        assert 0 <= result["k_percent"] <= 100
        assert 0 <= result["d_percent"] <= 100

    def test_signal_values(self, tech, synthetic_df):
        result = tech._calculate_stochastic(synthetic_df)
        assert result["signal"] in ("oversold", "overbought", "neutral")

    def test_crossover_values(self, tech, synthetic_df):
        result = tech._calculate_stochastic(synthetic_df)
        assert result["crossover"] in ("bullish", "bearish", "none")

    def test_insufficient_data(self, tech):
        short = pd.DataFrame(
            {"High": [10, 11], "Low": [9, 10], "Close": [9.5, 10.5]},
            index=pd.bdate_range("2025-01-01", periods=2),
        )
        assert tech._calculate_stochastic(short) == {}


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestVolumeProfile:
    def test_return_keys(self, tech, synthetic_df):
        result = tech._calculate_volume_profile(synthetic_df)
        for key in ("poc", "value_area_high", "value_area_low", "position"):
            assert key in result

    def test_value_area_ordering(self, tech, synthetic_df):
        result = tech._calculate_volume_profile(synthetic_df)
        assert result["value_area_low"] <= result["poc"] <= result["value_area_high"]

    def test_position_values(self, tech, synthetic_df):
        result = tech._calculate_volume_profile(synthetic_df)
        assert result["position"] in (
            "above_value_area", "below_value_area", "inside_value_area"
        )

    def test_no_volume_column(self, tech):
        df = pd.DataFrame(
            {"High": range(20), "Low": range(20), "Close": range(20)},
            index=pd.bdate_range("2025-01-01", periods=20),
        )
        assert tech._calculate_volume_profile(df) == {}

    def test_insufficient_data(self, tech):
        short = pd.DataFrame(
            {"High": [10], "Low": [9], "Close": [9.5], "Volume": [1000]},
            index=pd.bdate_range("2025-01-01", periods=1),
        )
        assert tech._calculate_volume_profile(short) == {}


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestFibonacci:
    def test_return_keys(self, tech, synthetic_df):
        result = tech._calculate_fibonacci_levels(synthetic_df)
        for key in ("levels", "swing_high", "swing_low", "current_price",
                     "closest_level", "closest_price", "distance_to_closest"):
            assert key in result

    def test_levels_between_swing(self, tech, synthetic_df):
        result = tech._calculate_fibonacci_levels(synthetic_df)
        for price in result["levels"].values():
            assert result["swing_low"] <= price <= result["swing_high"]

    def test_five_fib_levels(self, tech, synthetic_df):
        result = tech._calculate_fibonacci_levels(synthetic_df)
        assert len(result["levels"]) == 5

    def test_distance_non_negative(self, tech, synthetic_df):
        result = tech._calculate_fibonacci_levels(synthetic_df)
        assert result["distance_to_closest"] >= 0

    def test_flat_price_returns_empty(self, tech):
        flat = pd.DataFrame(
            {"High": [100.0] * 20, "Low": [100.0] * 20, "Close": [100.0] * 20},
            index=pd.bdate_range("2025-01-01", periods=20),
        )
        assert tech._calculate_fibonacci_levels(flat) == {}

    def test_insufficient_data(self, tech):
        short = pd.DataFrame(
            {"High": [10], "Low": [9], "Close": [9.5]},
            index=pd.bdate_range("2025-01-01", periods=1),
        )
        # With lookback=50 but min(lookback, 10) check, 1 row still works
        # but diff will be 1 so it should return data
        result = tech._calculate_fibonacci_levels(short)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Integration: calculate_all_indicators includes new keys
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
def test_calculate_all_indicators_includes_advanced(tech, synthetic_df):
    result = tech.calculate_all_indicators(synthetic_df)
    for key in ("adx", "atr", "stochastic", "volume_profile", "fibonacci"):
        assert key in result, f"calculate_all_indicators missing '{key}'"
