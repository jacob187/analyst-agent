"""Tests for session-company integration (Phase 4)."""

import pytest
from api.db import (
    init_db,
    create_session,
    get_or_create_session,
    get_company,
    get_company_activity,
    save_briefing,
)

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """Isolated SQLite DB for each test."""
    import api.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "_db", None)
    await init_db()
    yield
    await db_mod.close_db()


class TestSessionCreatesCompany:
    async def test_create_session_ensures_company(self, db):
        await create_session("AAPL", USER_A)
        company = await get_company("AAPL")
        assert company is not None
        assert company["ticker"] == "AAPL"

    async def test_get_or_create_session_ensures_company(self, db):
        session_id = await get_or_create_session("XOM", USER_A)
        assert session_id is not None
        company = await get_company("XOM")
        assert company is not None

    async def test_session_does_not_duplicate_company(self, db):
        await create_session("AAPL", USER_A)
        await create_session("AAPL", USER_A)
        # Should not raise — INSERT OR IGNORE handles duplicates


class TestCompanyActivity:
    async def test_empty_activity(self, db):
        from api.db import ensure_company
        await ensure_company("MSFT")
        activity = await get_company_activity("MSFT", USER_A)
        assert activity["ticker"] == "MSFT"
        assert activity["sessions"] == []
        assert activity["briefings"] == []

    async def test_activity_with_sessions(self, db):
        await create_session("AAPL", USER_A)
        await create_session("AAPL", USER_A)
        activity = await get_company_activity("AAPL", USER_A)
        assert len(activity["sessions"]) == 2

    async def test_activity_with_briefings(self, db):
        await save_briefing(
            raw_json="{}", market_regime="Bull", market_positioning="Long",
            alerts_json="[]", thinking=None,
            tickers=[{
                "ticker": "AAPL", "price": 180.0, "change_pct": 1.0,
                "technical_signal": "RSI", "news_summary": "News",
                "outlook": "bullish",
            }],
            user_id=USER_A,
        )
        activity = await get_company_activity("AAPL", USER_A)
        assert len(activity["briefings"]) == 1
        assert activity["briefings"][0]["outlook"] == "bullish"

    async def test_activity_mixed(self, db):
        await create_session("XOM", USER_A)
        await save_briefing(
            raw_json="{}", market_regime="Bear", market_positioning="Short",
            alerts_json="[]", thinking=None,
            tickers=[{
                "ticker": "XOM", "price": 110.0, "change_pct": -0.5,
                "technical_signal": "MACD", "news_summary": "Oil down",
                "outlook": "bearish",
            }],
            user_id=USER_A,
        )
        activity = await get_company_activity("XOM", USER_A)
        assert len(activity["sessions"]) == 1
        assert len(activity["briefings"]) == 1
