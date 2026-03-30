"""Market regime detection.

Analyzes SPY and VIX to determine the current market environment (trend,
volatility, Wyckoff phase) and generate strategy recommendations. This gives
individual stock analysis broader market context.
"""

from typing import Dict, Any, List

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


class MarketRegimeDetector:
    """Detect the current market regime from SPY and VIX data.

    Combines three lenses:
    1. Trend -- SPY MA50 vs MA200 relationship
    2. Volatility -- VIX absolute level
    3. Phase -- simplified Wyckoff cycle from price + volume trends
    """

    def detect_regime(self) -> Dict[str, Any]:
        """Fetch SPY and VIX, analyze each dimension, return composite regime.

        Returns:
            {trend, volatility, phase, recommendations: [...]}
        """
        spy_retriever = YahooFinanceDataRetrieval("SPY")
        vix_retriever = YahooFinanceDataRetrieval("^VIX")

        spy_hist = spy_retriever.get_historical_prices(period="1y", interval="1d")
        vix_hist = vix_retriever.get_historical_prices(period="3mo", interval="1d")

        trend = self._analyze_market_trend(spy_hist)
        volatility = self._analyze_volatility(vix_hist)
        phase = self._determine_market_phase(spy_hist)
        recommendations = self._generate_strategy_recommendations(trend, volatility, phase)

        return {
            "trend": trend,
            "volatility": volatility,
            "phase": phase,
            "recommendations": recommendations,
        }

    # -- Trend analysis ----------------------------------------------------

    def _analyze_market_trend(self, spy_hist) -> str:
        """Classify the broad market trend from SPY moving averages.

        - MA50 > MA200 and price above both -> bull
        - MA50 < MA200 and price below both -> bear
        - Otherwise -> transitional
        """
        if spy_hist is None or len(spy_hist) < 200:
            return "unknown"

        close = spy_hist["Close"]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        price = close.iloc[-1]

        if ma50 > ma200 and price > ma50:
            return "bull"
        elif ma50 < ma200 and price < ma50:
            return "bear"
        return "transitional"

    # -- Volatility analysis -----------------------------------------------

    def _analyze_volatility(self, vix_hist) -> str:
        """Map the current VIX level to a volatility regime.

        Thresholds:
            < 15  -> low
            < 20  -> moderate
            < 30  -> elevated
            >= 30 -> high
        """
        if vix_hist is None or vix_hist.empty:
            return "unknown"

        vix = float(vix_hist["Close"].iloc[-1])

        if vix < 15:
            return "low"
        elif vix < 20:
            return "moderate"
        elif vix < 30:
            return "elevated"
        return "high"

    # -- Market phase ------------------------------------------------------

    def _determine_market_phase(self, spy_hist) -> str:
        """Simplified Wyckoff phase from recent price trend + volume trend.

        Uses 50-bar lookback:
        - price_trend: positive or negative slope of a linear fit on close prices
        - volume_trend: whether recent 10-bar avg volume > 50-bar avg volume

        Mapping:
            price up   + volume up   -> markup
            price up   + volume down -> distribution
            price down + volume up   -> markdown
            price down + volume down -> accumulation
            flat                     -> consolidation
        """
        if spy_hist is None or len(spy_hist) < 50:
            return "unknown"

        close = spy_hist["Close"].iloc[-50:]
        volume = spy_hist["Volume"].iloc[-50:]

        # Price trend via simple slope
        x = range(len(close))
        price_slope = float(
            (len(close) * sum(i * c for i, c in zip(x, close.values))
             - sum(x) * sum(close.values))
            / (len(close) * sum(i * i for i in x) - sum(x) ** 2)
        )

        # Volume trend
        recent_vol = volume.iloc[-10:].mean()
        avg_vol = volume.mean()
        volume_rising = recent_vol > avg_vol

        # Classify
        price_pct = price_slope / close.mean() * 100
        if abs(price_pct) < 0.01:
            return "consolidation"
        elif price_pct > 0 and volume_rising:
            return "markup"
        elif price_pct > 0 and not volume_rising:
            return "distribution"
        elif price_pct < 0 and volume_rising:
            return "markdown"
        else:
            return "accumulation"

    # -- Strategy recommendations ------------------------------------------

    def _generate_strategy_recommendations(
        self, trend: str, volatility: str, phase: str
    ) -> List[str]:
        """Generate actionable strategy strings from the three regime dimensions."""
        recs: List[str] = []

        # Trend-based
        if trend == "bull":
            recs.append("Broad market is bullish - favor long positions and buy dips")
        elif trend == "bear":
            recs.append("Broad market is bearish - prioritize capital preservation and hedging")
        else:
            recs.append("Market in transition - reduce position sizes until direction clarifies")

        # Volatility-based
        if volatility == "low":
            recs.append("Low volatility regime - consider selling premium or trend strategies")
        elif volatility == "moderate":
            recs.append("Normal volatility - standard position sizing appropriate")
        elif volatility == "elevated":
            recs.append("Elevated volatility - tighten stops and reduce leverage")
        elif volatility == "high":
            recs.append("High volatility (VIX >= 30) - extreme caution, crisis-level risk management")

        # Phase-based
        phase_recs = {
            "markup": "Markup phase - momentum strategies work well, ride trends",
            "distribution": "Distribution phase - smart money may be selling, watch for reversals",
            "markdown": "Markdown phase - avoid catching falling knives, wait for capitulation",
            "accumulation": "Accumulation phase - early signs of bottoming, scale in gradually",
            "consolidation": "Consolidation phase - range-bound strategies (mean reversion) favored",
        }
        if phase in phase_recs:
            recs.append(phase_recs[phase])

        return recs
