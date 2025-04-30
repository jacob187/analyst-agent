from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import json

from agents.technical_agent.get_stock_data import YahooFinanceDataRetrieval
from agents.technical_agent.process_technical_indicators import TechnicalIndicators
from database.local_logger import LocalLogger


class TechnicalAgent:
    """Main technical agent that orchestrates the retrieval and analysis of stock data."""

    def __init__(self, ticker: str):
        """Initialize the technical agent for a specific ticker."""
        self.ticker = ticker
        self.data_retriever = YahooFinanceDataRetrieval(ticker)
        self.indicator_processor = TechnicalIndicators(ticker)
        self.logger = LocalLogger()

    def process_and_save(self) -> None:
        """
        Process the stock data and save the results.

        This is the main method that orchestrates the workflow:
        1. Retrieve stock data (prices, financials, info)
        2. Calculate technical indicators
        3. Save all data to storage
        """
        try:
            print(f"Starting technical analysis for {self.ticker}")

            # Step 1: Retrieve data
            print(f"Retrieving data for {self.ticker}...")
            # Get 3 months of historical data as a DataFrame
            hist_df = self.data_retriever.get_historical_prices(period="3mo")
            # Get financials and info (still saved to JSON by retriever)
            self.data_retriever.get_financials()
            self.data_retriever.get_info()

            # Check if we got price data
            if hist_df is None or hist_df.empty:
                print(
                    f"Skipping indicator calculation and analysis for {self.ticker} due to missing price data."
                )
                # Optionally, save a note that analysis failed or is incomplete
                data = self.logger.read_json()
                if self.ticker not in data:
                    data[self.ticker] = {}
                if "technical_agent" not in data[self.ticker]:
                    data[self.ticker]["technical_agent"] = {}
                data[self.ticker]["technical_agent"]["analysis"] = {
                    "summary": {"error": "Failed to retrieve historical price data"},
                    "timestamp": datetime.now().isoformat(),
                }
                self.logger.write_json(data)
                return  # Exit processing for this ticker

            # Step 2: Calculate indicators (pass DataFrame)
            print(f"Calculating technical indicators for {self.ticker}...")
            indicators = self.indicator_processor.calculate_all_indicators(hist_df)

            # Check if indicator calculation produced results
            if not indicators:
                print(
                    f"Skipping analysis summary for {self.ticker} due to empty indicators."
                )
                # Optionally save a note
                data = self.logger.read_json()
                if self.ticker not in data:
                    data[self.ticker] = {}
                if "technical_agent" not in data[self.ticker]:
                    data[self.ticker]["technical_agent"] = {}
                data[self.ticker]["technical_agent"]["analysis"] = {
                    "summary": {"error": "Failed to calculate technical indicators"},
                    "timestamp": datetime.now().isoformat(),
                }
                self.logger.write_json(data)
                return

            # Step 3: Generate technical analysis summary
            print(f"Generating technical analysis summary for {self.ticker}...")
            analysis_summary = self._generate_analysis_summary(indicators)

            # Step 4: Save ONLY the analysis summary
            data = self.logger.read_json()

            # Ensure ticker and keys exist
            data[self.ticker] = data.get(self.ticker, {})  # Ensure ticker key exists
            data[self.ticker]["technical_analysis"] = data[self.ticker].get(
                "technical_analysis", {}
            )  # Ensure technical_analysis key exists

            # Save the analysis summary under technical_analysis -> analysis
            data[self.ticker]["technical_analysis"]["analysis"] = {
                "summary": analysis_summary,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.write_json(data)
            print(f"Technical analysis for {self.ticker} completed and saved")

        except Exception as e:
            print(f"Error in technical agent processing: {e}")
            raise

    def get_analysis(self) -> Dict[str, Any]:
        """
        Get the technical analysis results for the ticker.

        Returns:
            A dictionary containing the technical analysis summary
        """
        data = self.logger.read_json()
        # Adjust path to retrieve the analysis summary
        analysis_data = (
            data.get(self.ticker, {}).get("technical_analysis", {}).get("analysis", {})
        )
        if not analysis_data:
            raise ValueError(
                f"No technical analysis summary found for {self.ticker}. Run process_and_save first."
            )
        return analysis_data

    def _generate_analysis_summary(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of technical analysis based on calculated indicators.

        Args:
            indicators: Dictionary of calculated technical indicators

        Returns:
            Dictionary containing technical analysis summary
        """
        summary = {
            "price_indicators": {},
            "trend_indicators": {},
            "volatility_indicators": {},
            "overall_signal": "",
        }

        # Price indicators
        if "moving_averages" in indicators:
            ma = indicators["moving_averages"]
            summary["price_indicators"]["current_price"] = ma["latest_close"]
            summary["price_indicators"]["ma_50"] = ma.get("MA_50", None)
            summary["price_indicators"]["ma_200"] = ma.get("MA_200", None)

            # Only check for golden cross if MA_200 exists
            if "MA_200" in ma and "trend_50_200" in ma:
                summary["price_indicators"]["golden_cross"] = (
                    ma["trend_50_200"] == "bullish"
                )
            else:
                summary["price_indicators"]["golden_cross"] = None

        # Bollinger Bands
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            summary["price_indicators"]["upper_band"] = bb["upper_band"]
            summary["price_indicators"]["lower_band"] = bb["lower_band"]
            summary["price_indicators"]["band_position"] = bb["position"]

        # Trend indicators
        if "rsi" in indicators:
            summary["trend_indicators"]["rsi"] = indicators["rsi"]["current"]
            summary["trend_indicators"]["rsi_signal"] = indicators["rsi"]["signal"]

        if "macd" in indicators:
            summary["trend_indicators"]["macd_signal"] = indicators["macd"]["signal"]
            summary["trend_indicators"]["macd_histogram"] = indicators["macd"][
                "histogram"
            ]

        # Volatility indicators
        if "volatility" in indicators:
            vol = indicators["volatility"]
            summary["volatility_indicators"]["daily_volatility"] = vol[
                "daily_volatility"
            ]
            summary["volatility_indicators"]["annualized_volatility"] = vol[
                "annualized_volatility"
            ]
            summary["volatility_indicators"]["max_drawdown"] = vol["max_drawdown"]

        # Financial metrics
        if "financial_metrics" in indicators:
            summary["financial_metrics"] = indicators["financial_metrics"]

        # Determine overall signal based on multiple indicators
        bullish_signals = 0
        bearish_signals = 0

        # Check MA trend
        if (
            "moving_averages" in indicators
            and "trend_50_200" in indicators["moving_averages"]
            and indicators["moving_averages"]["trend_50_200"] == "bullish"
        ):
            bullish_signals += 1
        elif (
            "moving_averages" in indicators
            and "trend_50_200" in indicators["moving_averages"]
        ):
            bearish_signals += 1

        # Check RSI
        if "rsi" in indicators:
            rsi_signal = indicators["rsi"]["signal"]
            if rsi_signal == "oversold":
                bullish_signals += 1
            elif rsi_signal == "overbought":
                bearish_signals += 1

        # Check MACD
        if "macd" in indicators:
            if indicators["macd"]["signal"] == "bullish":
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Determine overall signal
        if bullish_signals > bearish_signals:
            summary["overall_signal"] = "bullish"
        elif bearish_signals > bullish_signals:
            summary["overall_signal"] = "bearish"
        else:
            summary["overall_signal"] = "neutral"

        return summary


def main():
    """Main function to run the technical agent for a given ticker."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.technical_agent.main_technical_agent TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    agent = TechnicalAgent(ticker)
    agent.process_and_save()

    # Print summary
    analysis = agent.get_analysis()
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
