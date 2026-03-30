"""Tests for MultiTimeframeAnalyzer.

Tests the pure-logic methods (_determine_trend, _detect_conflicts,
_synthesize_recommendation) with mocked indicator dicts — no API calls needed.
"""

import pytest

from agents.technical_workflow.multi_timeframe import MultiTimeframeAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return MultiTimeframeAnalyzer("TEST")


# ---------------------------------------------------------------------------
# _determine_trend
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestDetermineTrend:
    def test_bullish_with_long_ma(self, analyzer):
        indicators = {
            "moving_averages": {"latest_close": 150, "MA_50": 140, "MA_200": 130}
        }
        assert analyzer._determine_trend(indicators) == "bullish"

    def test_bearish_with_long_ma(self, analyzer):
        indicators = {
            "moving_averages": {"latest_close": 100, "MA_50": 110, "MA_200": 120}
        }
        assert analyzer._determine_trend(indicators) == "bearish"

    def test_neutral_mixed(self, analyzer):
        # Price above MA50 but MA50 < MA200 -> neutral
        indicators = {
            "moving_averages": {"latest_close": 150, "MA_50": 140, "MA_200": 145}
        }
        assert analyzer._determine_trend(indicators) == "neutral"

    def test_fallback_short_ma_bullish(self, analyzer):
        indicators = {
            "moving_averages": {"latest_close": 50, "MA_5": 48, "MA_20": 45}
        }
        assert analyzer._determine_trend(indicators) == "bullish"

    def test_fallback_short_ma_bearish(self, analyzer):
        indicators = {
            "moving_averages": {"latest_close": 40, "MA_5": 42, "MA_20": 45}
        }
        assert analyzer._determine_trend(indicators) == "bearish"

    def test_empty_indicators(self, analyzer):
        assert analyzer._determine_trend({}) == "unknown"

    def test_empty_ma(self, analyzer):
        assert analyzer._determine_trend({"moving_averages": {}}) == "unknown"


# ---------------------------------------------------------------------------
# _detect_conflicts
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestDetectConflicts:
    def test_trend_conflict(self, analyzer):
        results = {
            "daily": {"trend": "bullish", "indicators": {"rsi": {"signal": "neutral"}}},
            "weekly": {"trend": "bearish", "indicators": {"rsi": {"signal": "neutral"}}},
        }
        conflicts = analyzer._detect_conflicts(results)
        trend_conflicts = [c for c in conflicts if c["type"] == "trend"]
        assert len(trend_conflicts) == 1
        assert "bullish" in trend_conflicts[0]["detail"]
        assert "bearish" in trend_conflicts[0]["detail"]

    def test_rsi_conflict(self, analyzer):
        results = {
            "daily": {"trend": "bullish", "indicators": {"rsi": {"signal": "overbought"}}},
            "1hr": {"trend": "bullish", "indicators": {"rsi": {"signal": "oversold"}}},
        }
        conflicts = analyzer._detect_conflicts(results)
        rsi_conflicts = [c for c in conflicts if c["type"] == "rsi"]
        assert len(rsi_conflicts) == 1

    def test_no_conflict_same_trend(self, analyzer):
        results = {
            "daily": {"trend": "bullish", "indicators": {"rsi": {"signal": "neutral"}}},
            "weekly": {"trend": "bullish", "indicators": {"rsi": {"signal": "neutral"}}},
        }
        assert analyzer._detect_conflicts(results) == []

    def test_error_timeframes_skipped(self, analyzer):
        results = {
            "daily": {"trend": "bullish", "indicators": {}},
            "weekly": {"error": "No data"},
        }
        # weekly has "error" key so it should be excluded
        assert analyzer._detect_conflicts(results) == []


# ---------------------------------------------------------------------------
# _synthesize_recommendation
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
class TestSynthesizeRecommendation:
    def test_all_bullish(self, analyzer):
        results = {
            "daily": {"trend": "bullish"},
            "weekly": {"trend": "bullish"},
            "1hr": {"trend": "bullish"},
        }
        rec = analyzer._synthesize_recommendation(results, [])
        assert rec["bias"] == "bullish"
        assert rec["confidence"] > 0.5
        assert rec["conflicts_present"] is False

    def test_all_bearish(self, analyzer):
        results = {
            "daily": {"trend": "bearish"},
            "weekly": {"trend": "bearish"},
            "1hr": {"trend": "bearish"},
        }
        rec = analyzer._synthesize_recommendation(results, [])
        assert rec["bias"] == "bearish"

    def test_mixed_signals_neutral(self, analyzer):
        results = {
            "daily": {"trend": "bullish"},
            "weekly": {"trend": "bearish"},
            "1hr": {"trend": "neutral"},
        }
        # With daily=+0.4, weekly=-0.1, 1hr=0 -> normalised = 0.3/0.8 = 0.375
        rec = analyzer._synthesize_recommendation(results, [])
        assert rec["bias"] in ("bullish", "neutral")

    def test_conflicts_reduce_confidence(self, analyzer):
        results = {
            "daily": {"trend": "bullish"},
            "weekly": {"trend": "bullish"},
            "1hr": {"trend": "bullish"},
        }
        no_conflict_rec = analyzer._synthesize_recommendation(results, [])
        conflicts = [{"type": "trend", "detail": "test"}]
        conflict_rec = analyzer._synthesize_recommendation(results, conflicts)
        assert conflict_rec["confidence"] < no_conflict_rec["confidence"]

    def test_empty_results(self, analyzer):
        rec = analyzer._synthesize_recommendation({}, [])
        assert rec["bias"] == "neutral"
        assert rec["confidence"] == 0.1  # min confidence

    def test_recommendation_keys(self, analyzer):
        rec = analyzer._synthesize_recommendation({}, [])
        for key in ("bias", "confidence", "score", "strategy", "conflicts_present"):
            assert key in rec
