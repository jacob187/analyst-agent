import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from database.local_logger import LocalLogger


class TechnicalIndicators:
    """Class to calculate technical indicators from stock data"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.logger = LocalLogger()

    def calculate_all_indicators(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate all technical indicators from the provided DataFrame.

        Args:
            df: DataFrame containing historical price data.

        Returns:
            Dictionary containing calculated indicators.
        """
        if df is None or df.empty:
            print(
                f"Warning: No price DataFrame provided for {self.ticker}. Skipping indicator calculation."
            )
            return {}

        # Load fundamental data separately if needed for financial metrics
        ticker_data = {}
        try:
            data = self.logger.read_json()
            if self.ticker in data and "technical_agent" in data[self.ticker]:
                ticker_data = data[self.ticker]["technical_agent"]
        except Exception as e:
            print(
                f"Could not load existing data for {self.ticker} for financial metrics: {e}"
            )

        # Calculate indicators
        indicators = {}

        # Moving averages
        indicators["moving_averages"] = self._calculate_moving_averages(df)

        # RSI
        indicators["rsi"] = self._calculate_rsi(df)

        # MACD
        indicators["macd"] = self._calculate_macd(df)

        # Bollinger Bands
        indicators["bollinger_bands"] = self._calculate_bollinger_bands(df)

        # Volatility
        indicators["volatility"] = self._calculate_volatility(df)

        # Growth metrics from financials if available
        if "financials" in ticker_data:
            indicators["financial_metrics"] = self._calculate_financial_metrics(
                ticker_data["financials"]
            )
        else:
            indicators["financial_metrics"] = {}

        return indicators

    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert dictionary data back to DataFrame"""
        if not data:  # Check if the dictionary is empty
            return pd.DataFrame()
        try:
            df = pd.DataFrame(
                data=data["data"],
                columns=data["columns"],
                index=pd.DatetimeIndex(data["index"]),
            )
            return df
        except Exception as e:
            print(f"Error converting dictionary to DataFrame: {e}")
            return pd.DataFrame()

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various moving averages"""
        close = df["Close"]
        result = {
            "MA_5": close.rolling(window=5).mean().iloc[-1],
            "MA_10": close.rolling(window=10).mean().iloc[-1],
            "MA_20": close.rolling(window=20).mean().iloc[-1],
            "MA_200": close.rolling(window=200).mean().iloc[-1],
            "latest_close": close.iloc[-1],
        }

        # Add MA_50 if we have enough data
        if len(close) >= 50:
            result["MA_50"] = close.rolling(window=50).mean().iloc[-1]

        # Add MA_200 and trend only if we have enough data
        if len(close) >= 200:
            result["MA_200"] = close.rolling(window=200).mean().iloc[-1]
            # Only calculate trend if we have both MAs
            if "MA_50" in result:
                result["trend_50_200"] = (
                    "bullish" if result["MA_50"] > result["MA_200"] else "bearish"
                )

        return result

    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index"""
        close = df["Close"]
        delta = close.diff()

        # Make two series: one for gains and one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        # Calculate the EWMA (Exponential Weighted Moving Average)
        roll_up = up.ewm(com=periods - 1, adjust=False).mean()
        roll_down = down.ewm(com=periods - 1, adjust=False).mean()

        # Calculate RS (Relative Strength)
        rs = roll_up / roll_down

        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))

        current_rsi = rsi.iloc[-1]

        return {
            "current": current_rsi,
            "signal": (
                "oversold"
                if current_rsi < 30
                else ("overbought" if current_rsi > 70 else "neutral")
            ),
        }

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        close = df["Close"]

        # Calculate EMAs
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_12 - ema_26

        # Calculate signal line (9-day EMA of MACD line)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        return {
            "macd_line": macd_line.iloc[-1],
            "signal_line": signal_line.iloc[-1],
            "histogram": histogram.iloc[-1],
            "signal": (
                "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"
            ),
        }

    def _calculate_bollinger_bands(
        self, df: pd.DataFrame, window: int = 20
    ) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        close = df["Close"]

        # Calculate rolling mean and standard deviation
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()

        # Calculate upper and lower bands
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        current_close = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = rolling_mean.iloc[-1]

        # Determine position relative to bands
        if current_close > current_upper:
            position = "above_upper"
        elif current_close < current_lower:
            position = "below_lower"
        else:
            position = "within_bands"

        return {
            "upper_band": current_upper,
            "middle_band": current_middle,
            "lower_band": current_lower,
            "position": position,
            "width": (current_upper - current_lower)
            / current_middle,  # Normalized width
        }

    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various volatility metrics"""
        # Daily returns
        returns = df["Close"].pct_change().dropna()

        volatility_daily = returns.std()
        volatility_annualized = volatility_daily * np.sqrt(
            252
        )  # Assuming 252 trading days in a year

        return {
            "daily_volatility": volatility_daily,
            "annualized_volatility": volatility_annualized,
            "max_drawdown": (df["Close"] / df["Close"].cummax() - 1.0).min(),
        }

    def _calculate_financial_metrics(
        self, financials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate key financial metrics from financial statements"""
        metrics = {}

        # Try to extract income statement data
        try:
            income_stmt = self._dict_to_dataframe(financials.get("income_stmt", {}))
            if not income_stmt.empty:
                # Calculate key growth rates
                total_revenue = (
                    income_stmt.loc["Total Revenue"]
                    if "Total Revenue" in income_stmt.index
                    else None
                )
                net_income = (
                    income_stmt.loc["Net Income"]
                    if "Net Income" in income_stmt.index
                    else None
                )

                if total_revenue is not None and len(total_revenue) >= 2:
                    metrics["revenue_growth_yoy"] = (
                        (total_revenue.iloc[0] / total_revenue.iloc[1]) - 1
                    ) * 100

                if net_income is not None and len(net_income) >= 2:
                    metrics["net_income_growth_yoy"] = (
                        (net_income.iloc[0] / net_income.iloc[1]) - 1
                    ) * 100
        except Exception as e:
            print(f"Error calculating income statement metrics: {e}")

        # Try to extract balance sheet data
        try:
            balance_sheet = self._dict_to_dataframe(financials.get("balance_sheet", {}))
            if not balance_sheet.empty:
                # Extract key metrics if available
                total_assets = (
                    balance_sheet.loc["Total Assets"]
                    if "Total Assets" in balance_sheet.index
                    else None
                )
                total_liabilities = (
                    balance_sheet.loc["Total Liabilities Net Minority Interest"]
                    if "Total Liabilities Net Minority Interest" in balance_sheet.index
                    else None
                )

                if total_assets is not None and total_liabilities is not None:
                    metrics["debt_to_assets"] = (
                        total_liabilities.iloc[0] / total_assets.iloc[0]
                    ) * 100
        except Exception as e:
            print(f"Error calculating balance sheet metrics: {e}")

        return metrics
