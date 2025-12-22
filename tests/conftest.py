"""
Pytest configuration and shared fixtures for analyst-agent tests.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime
import pandas as pd
from typing import Dict, Any


# ============================================================================
# Directory and File Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_data_json(temp_data_dir):
    """Create a mock data.json file for testing."""
    data_file = temp_data_dir / "data.json"
    sample_data = {
        "AAPL": {
            "sec_data": {"risk_factors": "Sample risk factors"},
            "technical_data": {"price": 150.0}
        }
    }
    with open(data_file, "w") as f:
        json.dump(sample_data, f)
    return data_file


# ============================================================================
# LLM Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for testing without making real API calls."""
    llm = MagicMock()
    llm.model_name = "gemini-2.5-flash-preview-05-20"

    # Mock invoke method to return structured responses
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "sentiment_score": 7,
        "key_risks": ["Risk 1", "Risk 2", "Risk 3"],
        "summary": "Test summary"
    })
    llm.invoke.return_value = mock_response

    return llm


@pytest.fixture
def mock_llm_with_structured_output():
    """Mock LLM with structured output for Pydantic models."""
    llm = MagicMock()

    def mock_with_structured_output(schema):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(
            sentiment_score=7,
            key_risks=["Risk 1", "Risk 2", "Risk 3"],
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            future_outlook="Positive"
        )
        return mock_chain

    llm.with_structured_output = mock_with_structured_output
    return llm


# ============================================================================
# SEC Data Fixtures
# ============================================================================

@pytest.fixture
def sample_sec_filing_metadata():
    """Sample SEC filing metadata."""
    return {
        "form": "10-K",
        "cik": "0000320193",
        "accession": "0000320193-24-000006",
        "filing_date": "2024-01-31",
        "period_of_report": "2023-09-30",
        "company_name": "Apple Inc."
    }


@pytest.fixture
def sample_risk_factors_text():
    """Sample risk factors text from 10-K filing."""
    return """
    ITEM 1A. RISK FACTORS

    Business and Operational Risks

    The Company's business can be impacted by political events, international trade disputes,
    war, terrorism, natural disasters, public health issues, and other business interruptions.

    The Company depends on component and product manufacturing and logistical services
    provided by outsourcing partners, many of which are located outside of the U.S.

    Financial Risks

    The Company is exposed to credit risk and fluctuations in the values of foreign currencies.
    """


@pytest.fixture
def sample_mda_text():
    """Sample Management Discussion and Analysis text."""
    return """
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

    The following discussion should be read in conjunction with the consolidated financial statements.

    Revenue increased by 15% year-over-year driven by strong iPhone and Services performance.
    Operating income improved due to better gross margins and operational efficiency.

    Looking forward, we expect continued growth in Services and wearables segments.
    """


@pytest.fixture
def sample_balance_sheet_data():
    """Sample balance sheet data in JSON format."""
    return {
        "tenk": {
            "columns": ["2023", "2022"],
            "index": ["Total Assets", "Total Liabilities", "Shareholders Equity"],
            "data": [
                [352755000000, 352583000000],
                [290437000000, 302083000000],
                [62318000000, 50500000000]
            ]
        },
        "tenq": {
            "columns": ["2024-Q1"],
            "index": ["Total Assets", "Total Liabilities", "Shareholders Equity"],
            "data": [
                [365725000000],
                [295000000000],
                [70725000000]
            ]
        }
    }


@pytest.fixture
def mock_sec_company():
    """Mock edgar Company object."""
    company = MagicMock()
    company.name = "Apple Inc."

    # Mock 10-K filing
    mock_10k_filing = MagicMock()
    mock_10k_filing.form = "10-K"
    mock_10k_filing.cik = "0000320193"
    mock_10k_filing.accession_number = "0000320193-24-000006"
    mock_10k_filing.filing_date = "2024-01-31"
    mock_10k_filing.period_of_report = "2023-09-30"
    mock_10k_filing.company = "Apple Inc."

    # Mock 10-K object
    mock_10k_obj = MagicMock()
    mock_10k_obj.risk_factors = "Sample risk factors text..."
    mock_10k_obj.management_discussion = "Sample MD&A text..."
    mock_10k_filing.obj.return_value = mock_10k_obj

    company.latest.return_value = mock_10k_filing

    return company


# ============================================================================
# Yahoo Finance Fixtures
# ============================================================================

@pytest.fixture
def sample_stock_prices():
    """Sample historical stock price data."""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
    df = pd.DataFrame({
        "Open": [150 + i * 0.5 for i in range(252)],
        "High": [152 + i * 0.5 for i in range(252)],
        "Low": [148 + i * 0.5 for i in range(252)],
        "Close": [151 + i * 0.5 for i in range(252)],
        "Volume": [50000000 + i * 100000 for i in range(252)],
    }, index=dates)
    return df


@pytest.fixture
def sample_financial_statements():
    """Sample financial statement data."""
    return {
        "income_stmt": {
            "columns": ["2023", "2022", "2021"],
            "index": ["Total Revenue", "Operating Income", "Net Income"],
            "data": [
                [383285000000, 394328000000, 365817000000],
                [114301000000, 119437000000, 108949000000],
                [96995000000, 99803000000, 94680000000]
            ]
        },
        "balance_sheet": {
            "columns": ["2023", "2022"],
            "index": ["Total Assets", "Total Liabilities"],
            "data": [
                [352755000000, 352583000000],
                [290437000000, 302083000000]
            ]
        }
    }


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object."""
    ticker = MagicMock()
    ticker.ticker = "AAPL"

    # Mock historical data
    dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
    ticker.history.return_value = pd.DataFrame({
        "Open": [150.0] * 252,
        "High": [152.0] * 252,
        "Low": [148.0] * 252,
        "Close": [151.0] * 252,
        "Volume": [50000000] * 252,
    }, index=dates)

    # Mock financials
    ticker.income_stmt = pd.DataFrame({
        "2023": [383285000000, 114301000000, 96995000000],
        "2022": [394328000000, 119437000000, 99803000000]
    }, index=["Total Revenue", "Operating Income", "Net Income"])

    ticker.balance_sheet = pd.DataFrame({
        "2023": [352755000000, 290437000000],
        "2022": [352583000000, 302083000000]
    }, index=["Total Assets", "Total Liabilities"])

    ticker.info = {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 2900000000000,
        "currency": "USD"
    }

    return ticker


# ============================================================================
# Technical Indicators Fixtures
# ============================================================================

@pytest.fixture
def sample_technical_indicators():
    """Sample technical indicators output."""
    return {
        "moving_averages": {
            "ma_5": 151.0,
            "ma_10": 150.5,
            "ma_20": 149.8,
            "ma_50": 148.0,
            "ma_200": 145.0
        },
        "rsi": {
            "value": 65.5,
            "signal": "neutral"
        },
        "macd": {
            "macd": 2.5,
            "signal": 2.0,
            "histogram": 0.5,
            "interpretation": "bullish"
        },
        "bollinger_bands": {
            "upper": 155.0,
            "middle": 150.0,
            "lower": 145.0,
            "position": "middle"
        },
        "volatility": {
            "daily": 0.02,
            "annualized": 0.32,
            "max_drawdown": -0.15
        }
    }


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-123")
    monkeypatch.setenv("SEC_HEADER", "test@example.com Test Company")


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset global caches between tests."""
    # Import here to avoid circular dependencies
    try:
        from agents.tools import sec_tools

        # Clear the caches
        sec_tools._shared_retrievers.clear()
        sec_tools._shared_processors.clear()
        sec_tools._processed_cache.clear()

        yield

        # Clear again after test
        sec_tools._shared_retrievers.clear()
        sec_tools._shared_processors.clear()
        sec_tools._processed_cache.clear()
    except ImportError:
        # If langchain is not installed, skip cache clearing
        yield


# ============================================================================
# MCP Server Fixtures (for future use)
# ============================================================================

@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing."""
    server = MagicMock()
    server.name = "yfinance-mcp"
    server.version = "1.0.0"

    # Mock tools list
    server.list_tools.return_value = [
        {"name": "get_stock_price", "description": "Get current stock price"},
        {"name": "get_historical_data", "description": "Get historical price data"},
        {"name": "get_financials", "description": "Get financial statements"}
    ]

    return server


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for testing."""
    client = MagicMock()

    # Mock call_tool method
    def mock_call_tool(tool_name, arguments):
        if tool_name == "get_stock_price":
            return {"price": 151.0, "currency": "USD"}
        elif tool_name == "get_historical_data":
            return {"data": [{"date": "2024-01-01", "close": 150.0}]}
        elif tool_name == "get_financials":
            return {"revenue": 383285000000}
        return {}

    client.call_tool = mock_call_tool
    return client
