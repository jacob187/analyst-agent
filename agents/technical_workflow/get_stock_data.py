import yfinance as yf
import pandas as pd
import json
from typing import Dict, Any, Optional


class YahooFinanceDataRetrieval:
    """Class to retrieve financial data using yfinance. Does NOT handle saving."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)

    def _dataframe_to_dict(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Convert a DataFrame to a JSON-serializable dictionary.
        Handles NaNs and Timestamps.

        Args:
            df: DataFrame to convert

        Returns:
            Dictionary representation of the DataFrame, or empty dict if input is None/empty.
        """
        if df is None or df.empty:
            return {}
        try:
            # Convert Timestamp index to string if it's a DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime("%Y-%m-%dT%H:%M:%S.%f")  # ISO format
            # Replace NaN/NaT with None for JSON compatibility
            df_serializable = df.where(pd.notnull(df), None)
            # Use pandas built-in JSON conversion which handles many types
            json_str = df_serializable.to_json(
                orient="split", date_format="iso", default_handler=str
            )
            return json.loads(json_str)
        except Exception as e:
            print(f"Error converting DataFrame to dict for {self.ticker}: {e}")
            return {}

    def get_financials(self) -> Dict[str, Any]:
        """
        Retrieve annual income statement and balance sheet from Yahoo Finance.

        Returns:
            Dict containing financial statement data, or empty dict on error.
        """
        try:
            income_stmt = self.yf_ticker.income_stmt
            balance_sheet = self.yf_ticker.balance_sheet
        except Exception as e:
            print(f"Error retrieving financial data for {self.ticker}: {e}")
            return {}

        # Convert DataFrames to JSON-serializable format
        financials = {
            "income_stmt": self._dataframe_to_dict(income_stmt),
            "balance_sheet": self._dataframe_to_dict(balance_sheet),
        }
        return financials  # Return fetched data, don't save

    def get_historical_prices(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical price data for the ticker.

        Args:
            period: Time period to retrieve (e.g., "1y", "3mo")

        Returns:
            DataFrame containing historical price data, or None if an error occurs.
        """
        try:
            hist = self.yf_ticker.history(period=period)
            if hist.empty:
                print(
                    f"Warning: No historical price data retrieved for {self.ticker} for period {period}"
                )
                return None
            return hist  # Return fetched data, don't save
        except Exception as e:
            print(f"Error retrieving historical prices for {self.ticker}: {e}")
            return None

    def get_info(self) -> Dict[str, Any]:
        """
        Get company information from Yahoo Finance.

        Returns:
            Dict containing company information, or empty dict on error.
        """
        try:
            info = self.yf_ticker.info
            # Clean up non-serializable types if necessary (though yfinance often handles this)
            serializable_info = {}
            for key, value in info.items():
                if (
                    isinstance(value, (int, float, str, bool, list, dict))
                    or value is None
                ):
                    serializable_info[key] = value
                elif isinstance(value, pd.Timestamp):
                    serializable_info[key] = value.isoformat()
                else:
                    # Attempt to convert other types to string, log if needed
                    try:
                        serializable_info[key] = str(value)
                    except:
                        print(
                            f"Warning: Could not serialize key '{key}' of type {type(value)} for {self.ticker}. Skipping."
                        )

            return serializable_info  # Return fetched data, don't save
        except Exception as e:
            print(f"Error retrieving company info for {self.ticker}: {e}")
            return {}


if __name__ == "__main__":
    retriever = YahooFinanceDataRetrieval("AAPL")
    info_data = retriever.get_info()
    financial_data = retriever.get_financials()
    hist_data = retriever.get_historical_prices(period="1y")

    if info_data:
        print("Info retrieved.")  # print(f"Info: {json.dumps(info_data, indent=2)}")
    if financial_data:
        print(
            "Financials retrieved."
        )  # print(f"Financials: {json.dumps(financial_data, indent=2)}")
    if hist_data is not None:
        print(f"Historical prices retrieved ({len(hist_data)} rows).")
