"""Multi-timeframe technical analysis.

Fetches data across daily, weekly, and hourly timeframes, calculates indicators
for each, detects conflicts between them, and synthesizes a weighted recommendation.
This helps identify when short-term momentum contradicts the longer-term trend.
"""

from typing import Dict, Any, List

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
from agents.technical_workflow.process_technical_indicators import TechnicalIndicators


class MultiTimeframeAnalyzer:
    """Analyze a stock across multiple timeframes and synthesize a recommendation.

    Timeframe weights (daily heaviest because most traders operate there):
        daily  = 0.40
        1hr    = 0.30
        weekly = 0.10
    Remaining 0.20 is reserved for conflict penalty / neutral.
    """

    TIMEFRAMES: Dict[str, Dict[str, str]] = {
        "daily": {"interval": "1d", "period": "1y"},
        "weekly": {"interval": "1wk", "period": "5y"},
        "1hr": {"interval": "1h", "period": "1mo"},
    }

    WEIGHTS: Dict[str, float] = {
        "daily": 0.40,
        "1hr": 0.30,
        "weekly": 0.10,
    }

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.retriever = YahooFinanceDataRetrieval(ticker)

    def analyze_all_timeframes(self) -> Dict[str, Any]:
        """Fetch data for each timeframe, compute indicators, and synthesize.

        Returns a dict with per-timeframe results, detected conflicts, and
        a final weighted recommendation.
        """
        results: Dict[str, Any] = {}
        tech = TechnicalIndicators(self.ticker)

        for name, params in self.TIMEFRAMES.items():
            try:
                hist = self.retriever.get_historical_prices(
                    period=params["period"], interval=params["interval"]
                )
                if hist is None or hist.empty:
                    results[name] = {"error": "No data available"}
                    continue

                indicators = tech.calculate_all_indicators(hist)
                trend = self._determine_trend(indicators)
                results[name] = {
                    "indicators": indicators,
                    "trend": trend,
                    "bars": len(hist),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        conflicts = self._detect_conflicts(results)
        recommendation = self._synthesize_recommendation(results, conflicts)

        return {
            "ticker": self.ticker,
            "timeframes": results,
            "conflicts": conflicts,
            "recommendation": recommendation,
        }

    # -- Trend classification ----------------------------------------------

    def _determine_trend(self, indicators: Dict[str, Any]) -> str:
        """Classify trend from moving-average relationships.

        Logic:
        - If MA50 > MA200 and price > MA50 -> bullish
        - If MA50 < MA200 and price < MA50 -> bearish
        - Otherwise -> neutral

        Falls back to shorter MAs (MA5 vs MA20) when 50/200 are unavailable
        (common for hourly data with limited history).
        """
        ma = indicators.get("moving_averages", {})
        if not ma:
            return "unknown"

        price = ma.get("latest_close", 0)

        # Prefer long-term MAs if available
        if "MA_50" in ma and "MA_200" in ma:
            if ma["MA_50"] > ma["MA_200"] and price > ma["MA_50"]:
                return "bullish"
            elif ma["MA_50"] < ma["MA_200"] and price < ma["MA_50"]:
                return "bearish"
            return "neutral"

        # Fallback to short-term MAs
        ma5 = ma.get("MA_5", 0)
        ma20 = ma.get("MA_20", 0)
        if ma5 and ma20:
            if price > ma5 > ma20:
                return "bullish"
            elif price < ma5 < ma20:
                return "bearish"
        return "neutral"

    # -- Conflict detection ------------------------------------------------

    def _detect_conflicts(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find trend or RSI conflicts between timeframes.

        A conflict exists when two timeframes disagree on direction (one bullish,
        the other bearish) or when RSI signals contradict across timeframes.
        """
        conflicts: List[Dict[str, str]] = []
        timeframe_names = [k for k in results if "error" not in results[k]]

        for i in range(len(timeframe_names)):
            for j in range(i + 1, len(timeframe_names)):
                tf_a = timeframe_names[i]
                tf_b = timeframe_names[j]
                trend_a = results[tf_a].get("trend", "unknown")
                trend_b = results[tf_b].get("trend", "unknown")

                # Trend conflict
                if (trend_a == "bullish" and trend_b == "bearish") or \
                   (trend_a == "bearish" and trend_b == "bullish"):
                    conflicts.append({
                        "type": "trend",
                        "timeframe_a": tf_a,
                        "timeframe_b": tf_b,
                        "detail": f"{tf_a} is {trend_a} but {tf_b} is {trend_b}",
                    })

                # RSI conflict
                rsi_a = results[tf_a].get("indicators", {}).get("rsi", {}).get("signal")
                rsi_b = results[tf_b].get("indicators", {}).get("rsi", {}).get("signal")
                if rsi_a and rsi_b and rsi_a != rsi_b and "neutral" not in (rsi_a, rsi_b):
                    conflicts.append({
                        "type": "rsi",
                        "timeframe_a": tf_a,
                        "timeframe_b": tf_b,
                        "detail": f"{tf_a} RSI is {rsi_a} but {tf_b} RSI is {rsi_b}",
                    })

        return conflicts

    # -- Recommendation synthesis ------------------------------------------

    def _synthesize_recommendation(
        self, results: Dict[str, Any], conflicts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Weighted scoring across timeframes.

        Each timeframe contributes a score: bullish=+1, bearish=-1, neutral/unknown=0,
        multiplied by its weight. Conflicts reduce overall confidence.
        """
        score = 0.0
        total_weight = 0.0
        trend_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "unknown": 0.0}

        for tf, weight in self.WEIGHTS.items():
            if tf in results and "error" not in results[tf]:
                trend = results[tf].get("trend", "unknown")
                score += trend_map.get(trend, 0.0) * weight
                total_weight += weight

        if total_weight > 0:
            normalised = score / total_weight  # range -1 to +1
        else:
            normalised = 0.0

        # Confidence drops with conflicts
        base_confidence = abs(normalised)
        conflict_penalty = len(conflicts) * 0.15
        confidence = max(0.1, base_confidence - conflict_penalty)

        if normalised > 0.2:
            bias = "bullish"
        elif normalised < -0.2:
            bias = "bearish"
        else:
            bias = "neutral"

        # Strategy suggestion
        if bias == "bullish" and confidence > 0.5:
            strategy = "Trend-following long positions favored"
        elif bias == "bearish" and confidence > 0.5:
            strategy = "Defensive positioning or short-term shorts favored"
        elif conflicts:
            strategy = "Mixed signals - reduce position size or wait for alignment"
        else:
            strategy = "No clear edge - monitor for breakout or breakdown"

        return {
            "bias": bias,
            "confidence": round(confidence, 2),
            "score": round(normalised, 3),
            "strategy": strategy,
            "conflicts_present": len(conflicts) > 0,
        }
