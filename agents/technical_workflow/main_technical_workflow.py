from typing import Dict, Any
from datetime import datetime
import json

from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval
from agents.technical_workflow.process_technical_indicators import TechnicalIndicators
from database.local_logger import LocalLogger


class TechnicalAgent:
    """Orchestrates retrieval, processing, and saving of technical stock data."""

    def __init__(self, ticker: str):
        """Initialize the technical agent for a specific ticker."""
        self.ticker = ticker
        # Instantiate components
        self.data_retriever = YahooFinanceDataRetrieval(ticker)
        self.indicator_processor = TechnicalIndicators(ticker)
        # Instantiate logger ONCE here
        self.logger = LocalLogger()

    def process_and_save(self) -> bool:
        """
        Orchestrates the workflow:
        1. Fetch all necessary raw data (1y prices, financials, info).
        2. Calculate technical indicators using 1y prices.
        3. Generate analysis summary.
        4. Read existing data from storage.
        5. Prepare the complete data structure for the ticker (incl. sliced 3mo prices).
        6. Save the updated structure back to storage.

        Returns:
            bool: True if processing and saving were successful, False otherwise.
        """
        try:
            print(f"Starting technical analysis orchestration for {self.ticker}")

            # --- Step 1: Fetch Raw Data ---
            print(f"Retrieving raw data for {self.ticker}...")
            info_data = self.data_retriever.get_info()
            financials_data = self.data_retriever.get_financials()
            hist_df_1y = self.data_retriever.get_historical_prices(period="1y")

            if hist_df_1y is None or hist_df_1y.empty:
                print(
                    f"ERROR: Failed to retrieve 1y historical price data for {self.ticker}. Aborting technical analysis."
                )
                return False  # <<< CRUCIAL: Return False on failure

            if not info_data:
                print(
                    f"Warning: Failed to retrieve company info for {self.ticker}. Proceeding without it."
                )
            if not financials_data:
                print(
                    f"Warning: Failed to retrieve financials data for {self.ticker}. Proceeding without it."
                )

            # --- Step 2: Calculate Indicators ---
            print(
                f"Calculating technical indicators for {self.ticker} using 1y data..."
            )
            indicators = self.indicator_processor.calculate_all_indicators(hist_df_1y)

            if not indicators:
                print(
                    f"ERROR: Indicator calculation failed for {self.ticker}. Aborting technical analysis."
                )
                return False  # <<< CRUCIAL: Return False on failure

            # --- Step 3: Generate Analysis Summary ---
            print(f"Generating technical analysis summary for {self.ticker}...")
            analysis_summary = self._generate_analysis_summary(indicators)

            # --- Step 4 & 5: Prepare Data and Save ---
            print(f"Preparing and saving all technical data for {self.ticker}...")
            try:
                # Read existing data ONCE
                all_data = self.logger.read_json()
            except FileNotFoundError:
                all_data = {}
                print("Data file not found, creating a new one.")

            # Get or create ticker entry
            ticker_data = all_data.setdefault(self.ticker, {})

            # --- Prepare technical_data section ---
            tech_data_section = ticker_data.setdefault("technical_data", {})
            tech_data_section["info"] = info_data
            tech_data_section["financials"] = financials_data

            # Slice the last 3 months (approx 63 trading days) for saving
            # Use .tail() for robustness if exact date slicing is complex
            hist_df_3mo = hist_df_1y.tail(63)  # Adjust number if needed
            tech_data_section["short_term_prices"] = (
                self.data_retriever._dataframe_to_dict(hist_df_3mo)
            )
            tech_data_section["last_updated"] = datetime.now().isoformat()

            # --- Prepare technical_analysis section ---
            tech_analysis_section = ticker_data.setdefault("technical_analysis", {})
            tech_analysis_section["analysis"] = {
                "summary": analysis_summary,
                "timestamp": datetime.now().isoformat(),
                "error": False,  # Indicate success
            }
            tech_analysis_section["last_updated"] = datetime.now().isoformat()

            # --- Step 6: Write updated data ONCE ---
            self.logger.write_json(all_data)

            print(
                f"Technical analysis orchestration for {self.ticker} completed and saved."
            )
            return True  # <<< CRUCIAL: Return True on success

        except Exception as e:
            print(f"Error during technical agent orchestration for {self.ticker}: {e}")

            return False

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

        return summary


def main():
    """Main function to run the technical agent for a given ticker."""
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m agents.technical_workflow.main_technical_workflow TICKER"
        )
        sys.exit(1)

    ticker = sys.argv[1].upper()
    agent = TechnicalAgent(ticker)

    success = agent.process_and_save()  # Capture the boolean result

    if success:
        # Only attempt to get analysis if processing was successful
        try:
            analysis = agent.get_analysis()
            print(f"\nTechnical Analysis Summary for {ticker}:")
            print(json.dumps(analysis, indent=2))
        except ValueError as e:
            # This case should be less common if 'success' is true, but good for safety
            print(
                f"Error retrieving analysis for {ticker} even after reported success: {e}"
            )
    else:
        # This message will now be printed if process_and_save returns False
        print(
            f"\nTechnical analysis for {ticker} could not be completed due to errors. No summary to display."
        )


if __name__ == "__main__":
    main()
