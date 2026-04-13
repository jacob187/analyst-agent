"""Tests for agents/tools/briefing_tools.py."""

from unittest.mock import patch, AsyncMock

from agents.tools.briefing_tools import (
    _tool_briefing_history,
    _tool_latest_briefing,
    create_briefing_tools,
)

USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


class TestBriefingHistoryTool:
    @patch("agents.tools.briefing_tools.asyncio.run")
    def test_returns_formatted_history(self, mock_run):
        mock_run.return_value = [
            {
                "created_at": "2026-04-01 08:00:00",
                "outlook": "bullish",
                "price": 180.50,
                "change_pct": 1.23,
                "technical_signal": "MACD bullish crossover",
                "news_summary": "Apple Q2 earnings beat",
            },
        ]
        result = _tool_briefing_history("AAPL", USER_ID, "7")
        assert "AAPL" in result
        assert "BULLISH" in result
        assert "180.50" in result
        assert "MACD bullish crossover" in result

    @patch("agents.tools.briefing_tools.asyncio.run")
    def test_no_history(self, mock_run):
        mock_run.return_value = []
        result = _tool_briefing_history("MSFT", USER_ID, "7")
        assert "No briefing history" in result
        assert "MSFT" in result

    @patch("agents.tools.briefing_tools.asyncio.run")
    def test_default_days(self, mock_run):
        mock_run.return_value = []
        _tool_briefing_history("AAPL", USER_ID, "invalid")
        # Should default to 7 days and not crash
        mock_run.assert_called_once()

    @patch("agents.tools.briefing_tools.asyncio.run")
    def test_multiple_entries(self, mock_run):
        mock_run.return_value = [
            {"created_at": "2026-04-01", "outlook": "bullish", "price": 180.0,
             "change_pct": 1.0, "technical_signal": "RSI oversold", "news_summary": "News 1"},
            {"created_at": "2026-03-31", "outlook": "bearish", "price": 178.0,
             "change_pct": -0.5, "technical_signal": "MACD bearish", "news_summary": "News 2"},
        ]
        result = _tool_briefing_history("AAPL", USER_ID)
        assert "BULLISH" in result
        assert "BEARISH" in result

    def test_no_user_id_returns_message(self):
        result = _tool_briefing_history("AAPL", None)
        assert "no user context" in result


class TestLatestBriefingTool:
    @patch("agents.tools.briefing_tools.asyncio.run")
    def test_returns_formatted_briefing(self, mock_run):
        mock_run.return_value = [
            {
                "created_at": "2026-04-01",
                "market_regime": "Bull, Low Volatility",
                "market_positioning": "Stay long, trail stops",
                "alerts": '["RSI divergence on XOM"]',
                "tickers": [
                    {"ticker": "AAPL", "price": 180.0, "change_pct": 1.0,
                     "outlook": "bullish", "technical_signal": "MACD cross",
                     "news_summary": "Earnings beat"},
                ],
            },
        ]
        result = _tool_latest_briefing(USER_ID)
        assert "Bull, Low Volatility" in result
        assert "AAPL" in result
        assert "RSI divergence on XOM" in result

    @patch("agents.tools.briefing_tools.asyncio.run")
    def test_no_briefings(self, mock_run):
        mock_run.return_value = []
        result = _tool_latest_briefing(USER_ID)
        assert "No briefings" in result


class TestCreateBriefingTools:
    def test_creates_two_tools(self):
        tools = create_briefing_tools("AAPL", user_id=USER_ID)
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "get_briefing_history" in names
        assert "get_latest_briefing" in names
