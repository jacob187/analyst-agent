import yfinance as yf
import pandas as pd
import json
from typing import Dict, Any, Optional

from database.local_logger import LocalLogger


class YahooFinanceDataRetrieval:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)
        self.logger = LocalLogger()

    def get_financials(self) -> Dict[str, Any]:
        """
        Retrieve financial statements from Yahoo Finance and save them using LocalLogger.

        Returns:
            Dict containing all financial data
        """
        # Retrieve financial data
        try:
            income_stmt = self.yf_ticker.income_stmt
            balance_sheet = self.yf_ticker.balance_sheet
            # cash_flow = self.yf_ticker.cashflow # Removed annual cash flow
            # quarterly_income = self.yf_ticker.quarterly_income_stmt # Removed quarterly
            # quarterly_balance = self.yf_ticker.quarterly_balance_sheet # Removed quarterly
            # quarterly_cash_flow = self.yf_ticker.quarterly_cashflow # Removed quarterly
        except Exception as e:
            print(f"Error retrieving financial data for {self.ticker}: {e}")
            return {}

        # Convert remaining DataFrames to JSON-serializable format
        financials = {
            "income_stmt": self._dataframe_to_dict(income_stmt),
            "balance_sheet": self._dataframe_to_dict(balance_sheet),
            # "cash_flow": self._dataframe_to_dict(cash_flow),
            # "quarterly_income_stmt": self._dataframe_to_dict(quarterly_income),
            # "quarterly_balance_sheet": self._dataframe_to_dict(quarterly_balance),
            # "quarterly_cashflow": self._dataframe_to_dict(quarterly_cash_flow),
        }

        # Read existing data from logger
        data = self.logger.read_json()

        # Add or update data for this ticker under 'technical_data'
        data[self.ticker] = data.get(self.ticker, {})  # Ensure ticker key exists
        data[self.ticker]["technical_data"] = data[self.ticker].get(
            "technical_data", {}
        )  # Ensure technical_data key exists
        data[self.ticker]["technical_data"]["financials"] = financials
        data[self.ticker]["technical_data"][
            "last_updated"
        ] = pd.Timestamp.now().isoformat()

        # Save updated data
        self.logger.write_json(data)

        return financials

    def get_historical_prices(self, period: str = "3mo") -> Optional[pd.DataFrame]:
        """
        Get historical price data for the ticker.

        Args:
            period: Time period to retrieve (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

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
            # We no longer save the raw price data here.
            # It will be processed in memory.
            return hist
        except Exception as e:
            print(f"Error retrieving historical prices for {self.ticker}: {e}")
            return None

    def get_info(self) -> Dict[str, Any]:
        """
        Get company information.

        Returns:
            Dict containing company information
        """
        try:
            info = self.yf_ticker.info

            # Read existing data from logger
            data = self.logger.read_json()

            # Add or update data for this ticker under 'technical_data'
            data[self.ticker] = data.get(self.ticker, {})  # Ensure ticker key exists
            data[self.ticker]["technical_data"] = data[self.ticker].get(
                "technical_data", {}
            )  # Ensure technical_data key exists
            data[self.ticker]["technical_data"]["info"] = info
            data[self.ticker]["technical_data"][
                "last_updated"
            ] = pd.Timestamp.now().isoformat()

            # Save updated data
            self.logger.write_json(data)

            return info
        except Exception as e:
            print(f"Error retrieving company info for {self.ticker}: {e}")
            return {}

    def _dataframe_to_dict(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Convert a DataFrame to a JSON-serializable dictionary.

        Args:
            df: DataFrame to convert

        Returns:
            Dictionary representation of the DataFrame
        """
        if df is None or df.empty:
            return {}

        try:
            return json.loads(df.to_json(date_format="iso", orient="split"))
        except Exception as e:
            print(f"Error converting DataFrame to dict: {e}")
            return {}
