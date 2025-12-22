"""
Unit tests for YahooFinanceDataRetrieval class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from agents.technical_workflow.get_stock_data import YahooFinanceDataRetrieval


class TestYahooFinanceDataRetrieval:
    """Test cases for Yahoo Finance data retrieval."""

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_initialization(self, mock_ticker_class):
        """Test YahooFinanceDataRetrieval initializes correctly."""
        mock_ticker_class.return_value = MagicMock()

        retriever = YahooFinanceDataRetrieval("AAPL")

        assert retriever.ticker == "AAPL"
        mock_ticker_class.assert_called_once_with("AAPL")

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_historical_prices_success(self, mock_ticker_class, sample_stock_prices):
        """Test successful retrieval of historical prices."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_stock_prices
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_historical_prices(period="1y")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 252
        assert "Close" in result.columns
        mock_ticker.history.assert_called_once_with(period="1y")

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_historical_prices_empty_data(self, mock_ticker_class):
        """Test handling of empty historical data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("INVALID")
        result = retriever.get_historical_prices(period="1y")

        assert result is None

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_historical_prices_api_error(self, mock_ticker_class):
        """Test handling of API errors."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("API Error")
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_historical_prices(period="1y")

        assert result is None

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_financials_success(self, mock_ticker_class, sample_financial_statements):
        """Test successful retrieval of financial statements."""
        mock_ticker = MagicMock()

        # Create DataFrames for financial statements
        income_stmt_df = pd.DataFrame(
            sample_financial_statements["income_stmt"]["data"],
            index=sample_financial_statements["income_stmt"]["index"],
            columns=sample_financial_statements["income_stmt"]["columns"],
        )
        balance_sheet_df = pd.DataFrame(
            sample_financial_statements["balance_sheet"]["data"],
            index=sample_financial_statements["balance_sheet"]["index"],
            columns=sample_financial_statements["balance_sheet"]["columns"],
        )

        mock_ticker.income_stmt = income_stmt_df
        mock_ticker.balance_sheet = balance_sheet_df
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_financials()

        assert result is not None
        assert "income_stmt" in result
        assert "balance_sheet" in result
        assert result["income_stmt"] != {}
        assert result["balance_sheet"] != {}

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_financials_error(self, mock_ticker_class):
        """Test handling of financials retrieval error."""
        mock_ticker = MagicMock()
        mock_ticker.income_stmt = None
        type(mock_ticker).income_stmt = property(
            lambda self: (_ for _ in ()).throw(Exception("API Error"))
        )
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_financials()

        assert result == {}

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_info_success(self, mock_ticker_class):
        """Test successful retrieval of company info."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 2900000000000,
            "currency": "USD",
        }
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_info()

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["longName"] == "Apple Inc."
        assert result["marketCap"] == 2900000000000

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_info_with_timestamp(self, mock_ticker_class):
        """Test info retrieval with Timestamp serialization."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "symbol": "AAPL",
            "lastUpdated": pd.Timestamp("2024-01-15"),
        }
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_info()

        assert result is not None
        assert "lastUpdated" in result
        assert isinstance(result["lastUpdated"], str)

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_get_info_error(self, mock_ticker_class):
        """Test handling of info retrieval error."""
        mock_ticker = MagicMock()
        type(mock_ticker).info = property(
            lambda self: (_ for _ in ()).throw(Exception("API Error"))
        )
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever.get_info()

        assert result == {}

    def test_dataframe_to_dict_empty(self):
        """Test conversion of empty DataFrame."""
        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever._dataframe_to_dict(None)

        assert result == {}

        empty_df = pd.DataFrame()
        result = retriever._dataframe_to_dict(empty_df)

        assert result == {}

    def test_dataframe_to_dict_with_nan(self):
        """Test DataFrame conversion handles NaN values."""
        df = pd.DataFrame(
            {"A": [1.0, float("nan"), 3.0], "B": [4.0, 5.0, float("nan")]}
        )

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever._dataframe_to_dict(df)

        assert result is not None
        # NaN values should be converted to None
        assert isinstance(result, dict)

    def test_dataframe_to_dict_with_datetime_index(self):
        """Test DataFrame conversion with DatetimeIndex."""
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"Close": [150.0, 151.0, 152.0]}, index=dates)

        retriever = YahooFinanceDataRetrieval("AAPL")
        result = retriever._dataframe_to_dict(df)

        assert result is not None
        assert isinstance(result, dict)
        # Index should be converted to ISO format strings
        assert "index" in result or "columns" in result

    @patch("agents.technical_workflow.get_stock_data.yf.Ticker")
    def test_multiple_period_requests(self, mock_ticker_class, sample_stock_prices):
        """Test retrieving historical data with different periods."""
        mock_ticker = MagicMock()

        def mock_history(period):
            if period == "1y":
                return sample_stock_prices
            elif period == "3mo":
                return sample_stock_prices.head(60)
            else:
                return pd.DataFrame()

        mock_ticker.history.side_effect = mock_history
        mock_ticker_class.return_value = mock_ticker

        retriever = YahooFinanceDataRetrieval("AAPL")

        # Test 1 year
        result_1y = retriever.get_historical_prices(period="1y")
        assert len(result_1y) == 252

        # Test 3 months
        result_3mo = retriever.get_historical_prices(period="3mo")
        assert len(result_3mo) == 60

        # Test invalid period
        result_invalid = retriever.get_historical_prices(period="invalid")
        assert result_invalid is None
