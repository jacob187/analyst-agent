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

    def get_historical_prices(self, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical price data for the ticker.

        Args:
            period: Time period to retrieve (e.g., "1y", "3mo", "5d")
            interval: Candle interval (e.g., "1d", "1h", "15m")

        Returns:
            DataFrame containing historical price data, or None if an error occurs.
        """
        try:
            hist = self.yf_ticker.history(period=period, interval=interval)
            if hist.empty:
                print(
                    f"Warning: No historical price data retrieved for {self.ticker} for period {period}"
                )
                return None
            return hist  # Return fetched data, don't save
        except Exception as e:
            print(f"Error retrieving historical prices for {self.ticker}: {e}")
            return None

    def get_live_price(self) -> Dict[str, Any]:
        """Get a fresh live quote by creating a new Ticker instance.

        The main self.yf_ticker caches data internally after first access,
        so reusing it for live price in a long-lived process returns stale
        values. This method creates a throwaway Ticker to guarantee a fresh
        network fetch from Yahoo Finance's quote endpoint.

        Returns:
            Dict with 'price', 'previousClose', 'change', 'changePercent',
            or empty dict on failure.
        """
        try:
            fresh_ticker = yf.Ticker(self.ticker)
            info = fresh_ticker.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")

            if price is None:
                # Last-resort fallback: fresh fast_info
                fi = fresh_ticker.fast_info
                price = float(fi["last_price"])
                prev_close = prev_close or float(fi["previous_close"])

            if price is None:
                return {}

            result: Dict[str, Any] = {"price": round(float(price), 2)}
            if prev_close is not None:
                prev_close = float(prev_close)
                result["previousClose"] = round(prev_close, 2)
                result["change"] = round(float(price) - prev_close, 2)
                result["changePercent"] = round(
                    (float(price) - prev_close) / prev_close * 100, 2
                )
            return result
        except Exception as e:
            print(f"Error fetching live price for {self.ticker}: {e}")
            return {}

    def get_company_profile(self) -> Dict[str, Any]:
        """Return a curated subset of company info for dashboard display.

        Picks only the fields relevant for a company overview card —
        avoids exposing the full 100+ field yfinance .info dict which
        contains unstable/undocumented keys.

        Returns:
            Dict with company metadata and key financial metrics,
            or empty dict on error.
        """
        _PROFILE_FIELDS = {
            "shortName", "sector", "industry", "country", "website",
            "longBusinessSummary", "fullTimeEmployees",
            "marketCap", "trailingPE", "forwardPE", "priceToBook",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "dividendYield", "beta",
        }
        try:
            info = self.yf_ticker.info
            return {k: info[k] for k in _PROFILE_FIELDS if k in info}
        except Exception as e:
            print(f"Error retrieving company profile for {self.ticker}: {e}")
            return {}

    def get_earnings_calendar(self) -> Dict[str, Any]:
        """Return upcoming earnings date and analyst estimates.

        yfinance's .calendar property is inconsistent — it returns a
        DataFrame for some tickers and a dict for others, and may be
        entirely absent. This method normalises all cases into a simple
        dict.

        Returns:
            Dict with 'earnings_date', 'earnings_average', 'revenue_average'
            (all optional), or empty dict if unavailable.
        """
        try:
            cal = self.yf_ticker.calendar
            if cal is None:
                return {}

            # yfinance sometimes returns a DataFrame, sometimes a dict
            if isinstance(cal, pd.DataFrame):
                cal = cal.to_dict()
                # DataFrame.to_dict() wraps values in {0: value} — unwrap
                # Guard against empty dicts which yfinance occasionally returns
                cal = {k: (list(v.values())[0] if isinstance(v, dict) and v else v)
                       for k, v in cal.items()}

            result: Dict[str, Any] = {}

            # Earnings date — may be a Timestamp, list of Timestamps, or string
            earnings_date = cal.get("Earnings Date")
            if isinstance(earnings_date, list) and earnings_date:
                earnings_date = earnings_date[0]
            if isinstance(earnings_date, pd.Timestamp):
                result["earnings_date"] = earnings_date.strftime("%Y-%m-%d")
            elif earnings_date is not None:
                result["earnings_date"] = str(earnings_date)

            for src_key, dst_key in [
                ("Earnings Average", "earnings_average"),
                ("Revenue Average", "revenue_average"),
                ("Earnings High", "earnings_high"),
                ("Earnings Low", "earnings_low"),
            ]:
                val = cal.get(src_key)
                if val is not None:
                    try:
                        result[dst_key] = round(float(val), 4)
                    except (TypeError, ValueError):
                        pass

            return result
        except Exception as e:
            print(f"Error retrieving earnings calendar for {self.ticker}: {e}")
            return {}

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
