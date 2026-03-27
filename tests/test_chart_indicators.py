"""Tests for TechnicalIndicators chart time series output.

Verifies that:
1. calculate_chart_indicators() returns properly formatted time series
2. calculate_all_indicators() output is unchanged after refactor (regression)
3. Edge cases: empty DataFrames, small DataFrames
"""

import numpy as np
import pandas as pd
import pytest

from agents.technical_workflow.process_technical_indicators import TechnicalIndicators


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def make_ohlcv_df():
    """Factory that builds a synthetic OHLCV DataFrame of any length.

    Generates a random walk starting at price=100 with realistic OHLCV
    columns and a DatetimeIndex.
    """

    def _make(rows: int = 252) -> pd.DataFrame:
        rng = np.random.default_rng(42)  # deterministic seed
        dates = pd.bdate_range(end="2026-03-27", periods=rows)

        # Random walk for close prices
        returns = rng.normal(0.0005, 0.02, size=rows)
        close = 100 * np.cumprod(1 + returns)

        # Derive OHLV from close
        high = close * (1 + rng.uniform(0, 0.02, size=rows))
        low = close * (1 - rng.uniform(0, 0.02, size=rows))
        open_ = close * (1 + rng.uniform(-0.01, 0.01, size=rows))
        volume = rng.integers(1_000_000, 10_000_000, size=rows)

        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    return _make


@pytest.fixture
def df_1y(make_ohlcv_df):
    """252-row DataFrame (simulates 1 year of trading days)."""
    return make_ohlcv_df(252)


@pytest.fixture
def df_small(make_ohlcv_df):
    """30-row DataFrame (not enough for MA_200 or MA_50)."""
    return make_ohlcv_df(30)


@pytest.fixture
def ti():
    """TechnicalIndicators instance."""
    return TechnicalIndicators("TEST")


# ── Regression: calculate_all_indicators output shape ─────────────────────


class TestCalculateAllIndicatorsRegression:
    """Ensure the agent-facing method returns the exact same dict shape."""

    def test_returns_expected_top_level_keys(self, ti, df_1y):
        result = ti.calculate_all_indicators(df_1y)
        assert set(result.keys()) == {
            "moving_averages", "rsi", "macd", "bollinger_bands", "volatility"
        }

    def test_moving_averages_keys(self, ti, df_1y):
        ma = ti.calculate_all_indicators(df_1y)["moving_averages"]
        assert "MA_5" in ma
        assert "MA_10" in ma
        assert "MA_20" in ma
        assert "MA_50" in ma
        assert "MA_200" in ma
        assert "latest_close" in ma
        assert "trend_50_200" in ma

    def test_moving_averages_values_are_floats(self, ti, df_1y):
        ma = ti.calculate_all_indicators(df_1y)["moving_averages"]
        for key in ["MA_5", "MA_10", "MA_20", "MA_50", "MA_200", "latest_close"]:
            assert isinstance(float(ma[key]), float)

    def test_rsi_keys(self, ti, df_1y):
        rsi = ti.calculate_all_indicators(df_1y)["rsi"]
        assert set(rsi.keys()) == {"current", "signal"}
        assert isinstance(float(rsi["current"]), float)
        assert rsi["signal"] in ("oversold", "overbought", "neutral")

    def test_macd_keys(self, ti, df_1y):
        macd = ti.calculate_all_indicators(df_1y)["macd"]
        assert set(macd.keys()) == {"macd_line", "signal_line", "histogram", "signal"}
        assert macd["signal"] in ("bullish", "bearish")

    def test_bollinger_keys(self, ti, df_1y):
        bb = ti.calculate_all_indicators(df_1y)["bollinger_bands"]
        assert set(bb.keys()) == {
            "upper_band", "middle_band", "lower_band", "position", "width"
        }
        assert bb["position"] in ("above_upper", "below_lower", "within_bands")

    def test_volatility_keys(self, ti, df_1y):
        vol = ti.calculate_all_indicators(df_1y)["volatility"]
        assert set(vol.keys()) == {
            "daily_volatility", "annualized_volatility", "max_drawdown"
        }

    def test_small_df_omits_ma200(self, ti, df_small):
        """With only 30 rows, MA_50 and MA_200 should be absent."""
        ma = ti.calculate_all_indicators(df_small)["moving_averages"]
        assert "MA_200" not in ma
        assert "MA_50" not in ma
        assert "trend_50_200" not in ma

    def test_empty_df_returns_empty(self, ti):
        result = ti.calculate_all_indicators(pd.DataFrame())
        assert result == {}

    def test_none_df_returns_empty(self, ti):
        result = ti.calculate_all_indicators(None)
        assert result == {}


# ── Chart indicators: time series format ──────────────────────────────────


class TestCalculateChartIndicators:
    """Verify the chart-facing method returns properly formatted time series."""

    def test_returns_expected_keys_1y(self, ti, df_1y):
        result = ti.calculate_chart_indicators(df_1y)
        assert "ma20" in result
        assert "ma50" in result
        assert "ma200" in result
        assert "rsi" in result
        assert "macd" in result
        assert "bollinger" in result

    def test_small_df_omits_long_mas(self, ti, df_small):
        result = ti.calculate_chart_indicators(df_small)
        assert "ma20" in result
        assert "ma50" not in result
        assert "ma200" not in result

    def test_empty_df_returns_empty(self, ti):
        assert ti.calculate_chart_indicators(pd.DataFrame()) == {}

    def test_none_df_returns_empty(self, ti):
        assert ti.calculate_chart_indicators(None) == {}

    # ── MA series format ──

    def test_ma_series_format(self, ti, df_1y):
        ma20 = ti.calculate_chart_indicators(df_1y)["ma20"]
        assert isinstance(ma20, list)
        assert len(ma20) > 0
        first = ma20[0]
        assert "time" in first
        assert "value" in first
        assert isinstance(first["time"], str)
        assert isinstance(first["value"], float)

    def test_ma_series_excludes_nan(self, ti, df_1y):
        """MA_20 needs 20 warmup bars, so series should be shorter than input."""
        ma20 = ti.calculate_chart_indicators(df_1y)["ma20"]
        assert len(ma20) < len(df_1y)
        # Specifically, MA_20 should have ~232 entries for 252 rows
        assert len(ma20) == 252 - 19  # rolling(20) produces NaN for first 19

    def test_ma_series_length_ordering(self, ti, df_1y):
        """Longer MAs should produce shorter series (more warmup needed)."""
        result = ti.calculate_chart_indicators(df_1y)
        assert len(result["ma20"]) > len(result["ma50"]) > len(result["ma200"])

    # ── RSI series format ──

    def test_rsi_series_format(self, ti, df_1y):
        rsi = ti.calculate_chart_indicators(df_1y)["rsi"]
        assert isinstance(rsi, list)
        assert len(rsi) > 0
        first = rsi[0]
        assert "time" in first
        assert "value" in first

    def test_rsi_values_in_range(self, ti, df_1y):
        """RSI must be between 0 and 100."""
        rsi = ti.calculate_chart_indicators(df_1y)["rsi"]
        for point in rsi:
            assert 0 <= point["value"] <= 100

    # ── MACD series format ──

    def test_macd_series_format(self, ti, df_1y):
        macd = ti.calculate_chart_indicators(df_1y)["macd"]
        assert isinstance(macd, list)
        assert len(macd) > 0
        first = macd[0]
        assert set(first.keys()) == {"time", "macd", "signal", "histogram"}

    # ── Bollinger series format ──

    def test_bollinger_series_format(self, ti, df_1y):
        bb = ti.calculate_chart_indicators(df_1y)["bollinger"]
        assert isinstance(bb, list)
        assert len(bb) > 0
        first = bb[0]
        assert set(first.keys()) == {"time", "upper", "middle", "lower"}

    def test_bollinger_band_ordering(self, ti, df_1y):
        """Upper band should always be above middle, middle above lower."""
        bb = ti.calculate_chart_indicators(df_1y)["bollinger"]
        for point in bb:
            assert point["upper"] >= point["middle"] >= point["lower"]

    # ── Time format ──

    def test_time_format_is_iso_date(self, ti, df_1y):
        """All time values should be YYYY-MM-DD strings."""
        result = ti.calculate_chart_indicators(df_1y)
        for series_key in ["ma20", "rsi"]:
            for point in result[series_key]:
                # Validate format by parsing
                pd.Timestamp(point["time"])
                assert len(point["time"]) == 10  # "YYYY-MM-DD"

    # ── Consistency between methods ──

    def test_rsi_current_matches_series_last(self, ti, df_1y):
        """The agent-facing current RSI should match the last chart series value."""
        current = ti.calculate_all_indicators(df_1y)["rsi"]["current"]
        series = ti.calculate_chart_indicators(df_1y)["rsi"]
        last_chart_value = series[-1]["value"]
        assert abs(current - last_chart_value) < 0.01

    def test_macd_current_matches_series_last(self, ti, df_1y):
        """MACD current values should match last chart series entry."""
        current = ti.calculate_all_indicators(df_1y)["macd"]
        series = ti.calculate_chart_indicators(df_1y)["macd"]
        last = series[-1]
        assert abs(current["macd_line"] - last["macd"]) < 0.01
        assert abs(current["signal_line"] - last["signal"]) < 0.01
        assert abs(current["histogram"] - last["histogram"]) < 0.01

    # ── Rounding ──

    def test_values_rounded_to_4_decimals(self, ti, df_1y):
        """All numeric values in chart output should have at most 4 decimal places."""
        result = ti.calculate_chart_indicators(df_1y)
        # Check MA values
        for point in result["ma20"]:
            # Round-trip: round(val, 4) should equal val exactly
            assert point["value"] == round(point["value"], 4)
        # Check MACD multi-field values
        for point in result["macd"]:
            assert point["macd"] == round(point["macd"], 4)
            assert point["signal"] == round(point["signal"], 4)
            assert point["histogram"] == round(point["histogram"], 4)
        # Check Bollinger multi-field values
        for point in result["bollinger"]:
            assert point["upper"] == round(point["upper"], 4)
            assert point["middle"] == round(point["middle"], 4)
            assert point["lower"] == round(point["lower"], 4)


# ── Static chart format helpers (direct unit tests) ──────────────────────


class TestSeriesToChartFormat:
    """Direct tests for _series_to_chart_format static method."""

    def test_drops_nan_entries(self):
        """NaN values from rolling warmup should not appear in output."""
        index = pd.bdate_range(start="2026-01-01", periods=5)
        series = pd.Series([np.nan, np.nan, 10.0, 20.0, 30.0])
        result = TechnicalIndicators("TEST")._series_to_chart_format(series, index)
        assert len(result) == 3
        assert all("value" in r for r in result)

    def test_drops_inf_entries(self):
        """Infinite values should also be excluded."""
        index = pd.bdate_range(start="2026-01-01", periods=3)
        series = pd.Series([np.inf, -np.inf, 42.0])
        result = TechnicalIndicators("TEST")._series_to_chart_format(series, index)
        assert len(result) == 1
        assert result[0]["value"] == 42.0

    def test_output_shape(self):
        index = pd.bdate_range(start="2026-01-01", periods=3)
        series = pd.Series([1.12345, 2.6789, 3.0])
        result = TechnicalIndicators("TEST")._series_to_chart_format(series, index)
        assert len(result) == 3
        assert set(result[0].keys()) == {"time", "value"}
        # Values should be rounded to 4 decimals (pandas uses banker's rounding)
        assert result[0]["value"] == 1.1234


class TestMacdToChartFormat:
    """Direct tests for _macd_to_chart_format static method."""

    def test_produces_correct_fields(self):
        index = pd.bdate_range(start="2026-01-01", periods=3)
        macd = pd.Series([0.5, 1.0, 1.5])
        signal = pd.Series([0.3, 0.8, 1.2])
        hist = pd.Series([0.2, 0.2, 0.3])
        result = TechnicalIndicators("TEST")._macd_to_chart_format(macd, signal, hist, index)
        assert len(result) == 3
        assert set(result[0].keys()) == {"time", "macd", "signal", "histogram"}

    def test_drops_rows_with_any_nan(self):
        """If any of the three series has NaN at index i, that row is dropped."""
        index = pd.bdate_range(start="2026-01-01", periods=3)
        macd = pd.Series([np.nan, 1.0, 1.5])
        signal = pd.Series([0.3, np.nan, 1.2])
        hist = pd.Series([0.2, 0.2, 0.3])
        result = TechnicalIndicators("TEST")._macd_to_chart_format(macd, signal, hist, index)
        # Only the third row has all three valid
        assert len(result) == 1
        assert result[0]["time"] == index[2].strftime("%Y-%m-%d")


class TestBollingerToChartFormat:
    """Direct tests for _bollinger_to_chart_format static method."""

    def test_produces_correct_fields(self):
        index = pd.bdate_range(start="2026-01-01", periods=3)
        upper = pd.Series([110.0, 112.0, 115.0])
        middle = pd.Series([100.0, 101.0, 103.0])
        lower = pd.Series([90.0, 90.0, 91.0])
        result = TechnicalIndicators("TEST")._bollinger_to_chart_format(upper, middle, lower, index)
        assert len(result) == 3
        assert set(result[0].keys()) == {"time", "upper", "middle", "lower"}

    def test_drops_rows_with_any_nan(self):
        index = pd.bdate_range(start="2026-01-01", periods=3)
        upper = pd.Series([np.nan, 112.0, 115.0])
        middle = pd.Series([100.0, np.nan, 103.0])
        lower = pd.Series([90.0, 90.0, 91.0])
        result = TechnicalIndicators("TEST")._bollinger_to_chart_format(upper, middle, lower, index)
        assert len(result) == 1
        assert result[0]["time"] == index[2].strftime("%Y-%m-%d")
