"""Tests for agents/tools/briefing_tools.py — async tools over a real temp DB."""

import json

import pytest

import api.db as db_module
from agents.tools.briefing_tools import (
    _tool_briefing_history,
    _tool_latest_briefing,
    create_briefing_tools,
)

USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
OTHER_USER = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield
    await db_module.close_db()


def _ticker(ticker="AAPL", price=180.50, change=1.23, outlook="bullish",
            signal="MACD bullish crossover", news="Apple Q2 earnings beat"):
    return {
        "ticker": ticker, "price": price, "change_pct": change, "outlook": outlook,
        "technical_signal": signal, "news_summary": news,
    }


async def _save(tickers, user_id=USER_ID, alerts=("RSI divergence on XOM",)):
    return await db_module.save_briefing(
        raw_json="{}",
        market_regime="Bull, Low Volatility",
        market_positioning="Stay long, trail stops",
        alerts_json=json.dumps(list(alerts)),
        thinking=None,
        tickers=tickers,
        user_id=user_id,
    )


class TestBriefingHistoryTool:
    async def test_returns_formatted_history(self, fresh_db):
        await _save([_ticker()])
        result = await _tool_briefing_history("AAPL", USER_ID, "7")
        assert "AAPL" in result
        assert "BULLISH" in result
        assert "180.50" in result
        assert "MACD bullish crossover" in result

    async def test_no_history(self, fresh_db):
        result = await _tool_briefing_history("MSFT", USER_ID, "7")
        assert "No briefing history" in result
        assert "MSFT" in result

    async def test_invalid_days_defaults_to_seven(self, fresh_db):
        await _save([_ticker()])
        result = await _tool_briefing_history("AAPL", USER_ID, "invalid")
        assert "last 7 days" in result

    async def test_multiple_entries(self, fresh_db):
        await _save([_ticker(outlook="bearish", signal="MACD bearish")])
        await _save([_ticker(outlook="bullish", signal="RSI oversold")])
        result = await _tool_briefing_history("AAPL", USER_ID)
        assert "BULLISH" in result
        assert "BEARISH" in result

    async def test_scoped_to_user(self, fresh_db):
        await _save([_ticker()], user_id=OTHER_USER)
        result = await _tool_briefing_history("AAPL", USER_ID, "7")
        assert "No briefing history" in result

    async def test_no_user_id_returns_message(self):
        result = await _tool_briefing_history("AAPL", None)
        assert "no user context" in result


class TestLatestBriefingTool:
    async def test_returns_formatted_briefing(self, fresh_db):
        await _save([_ticker(price=180.0, change=1.0, signal="MACD cross", news="Earnings beat")])
        result = await _tool_latest_briefing(USER_ID)
        assert "Bull, Low Volatility" in result
        assert "AAPL" in result
        assert "RSI divergence on XOM" in result

    async def test_no_briefings(self, fresh_db):
        result = await _tool_latest_briefing(USER_ID)
        assert "No briefings" in result


class TestCreateBriefingTools:
    def test_creates_two_tools(self):
        tools = create_briefing_tools("AAPL", user_id=USER_ID)
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "get_briefing_history" in names
        assert "get_latest_briefing" in names

    async def test_tools_run_via_ainvoke_on_the_running_loop(self, fresh_db):
        await _save([_ticker()])
        tools = {t.name: t for t in create_briefing_tools("AAPL", user_id=USER_ID)}
        assert all(t.func is None and t.coroutine is not None for t in tools.values())

        history = await tools["get_briefing_history"].ainvoke("7")
        latest = await tools["get_latest_briefing"].ainvoke("")

        assert "BULLISH" in history
        assert "Bull, Low Volatility" in latest
