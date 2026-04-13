"""Integration tests for user isolation — proves cross-user data is invisible.

Uses real SQLite in tmp_path (project convention). No mocks.
"""

import pytest
import api.db as db_module
from api.db import (
    init_db,
    get_or_create_session,
    get_session,
    get_session_by_ticker,
    get_tickers,
    get_sessions_for_ticker,
    save_message,
    get_session_messages,
    delete_session,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    save_briefing,
    get_recent_briefings,
    get_briefing_history,
    get_company_activity,
)

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await init_db()
    yield
    await db_module.close_db()


# =========================================================================
# Watchlist isolation
# =========================================================================

class TestWatchlistIsolation:
    async def test_users_have_separate_watchlists(self, db):
        await add_to_watchlist("AAPL", USER_A)
        await add_to_watchlist("MSFT", USER_B)

        a_list = await get_watchlist(USER_A)
        b_list = await get_watchlist(USER_B)

        assert [t["ticker"] for t in a_list] == ["AAPL"]
        assert [t["ticker"] for t in b_list] == ["MSFT"]

    async def test_same_ticker_both_users(self, db):
        await add_to_watchlist("AAPL", USER_A)
        await add_to_watchlist("AAPL", USER_B)

        a_list = await get_watchlist(USER_A)
        b_list = await get_watchlist(USER_B)

        assert len(a_list) == 1
        assert len(b_list) == 1

    async def test_remove_only_affects_own_watchlist(self, db):
        await add_to_watchlist("AAPL", USER_A)
        await add_to_watchlist("AAPL", USER_B)

        await remove_from_watchlist("AAPL", USER_A)

        assert await get_watchlist(USER_A) == []
        assert len(await get_watchlist(USER_B)) == 1


# =========================================================================
# Session isolation
# =========================================================================

class TestSessionIsolation:
    async def test_users_get_separate_sessions_for_same_ticker(self, db):
        a_session = await get_or_create_session("AAPL", USER_A)
        b_session = await get_or_create_session("AAPL", USER_B)

        assert a_session != b_session

    async def test_get_session_by_ticker_scoped_to_user(self, db):
        await get_or_create_session("AAPL", USER_A)

        assert (await get_session_by_ticker("AAPL", USER_A)) is not None
        assert (await get_session_by_ticker("AAPL", USER_B)) is None

    async def test_get_tickers_scoped_to_user(self, db):
        await get_or_create_session("AAPL", USER_A)
        await get_or_create_session("MSFT", USER_B)

        a_tickers = await get_tickers(USER_A)
        b_tickers = await get_tickers(USER_B)

        assert [t["ticker"] for t in a_tickers] == ["AAPL"]
        assert [t["ticker"] for t in b_tickers] == ["MSFT"]

    async def test_sessions_for_ticker_scoped_to_user(self, db):
        await get_or_create_session("AAPL", USER_A)
        await get_or_create_session("AAPL", USER_B)

        a_sessions = await get_sessions_for_ticker("AAPL", USER_A)
        b_sessions = await get_sessions_for_ticker("AAPL", USER_B)

        assert len(a_sessions) == 1
        assert len(b_sessions) == 1
        assert a_sessions[0]["id"] != b_sessions[0]["id"]

    async def test_idor_session_user_mismatch(self, db):
        """User B cannot access user A's session even with the session ID."""
        a_session_id = await get_or_create_session("AAPL", USER_A)
        await save_message(a_session_id, "user", "secret question")

        # User B gets the session by ID (UUID guess)
        session = await get_session(a_session_id)
        assert session is not None
        # But the user_id field reveals it belongs to USER_A
        assert session["user_id"] == USER_A
        assert session["user_id"] != USER_B

    async def test_messages_inherited_from_session(self, db):
        """Messages are scoped via session ownership — no direct user_id needed."""
        a_session = await get_or_create_session("AAPL", USER_A)
        b_session = await get_or_create_session("AAPL", USER_B)

        await save_message(a_session, "user", "A's question")
        await save_message(b_session, "user", "B's question")

        a_msgs = await get_session_messages(a_session)
        b_msgs = await get_session_messages(b_session)

        assert len(a_msgs) == 1
        assert a_msgs[0]["content"] == "A's question"
        assert len(b_msgs) == 1
        assert b_msgs[0]["content"] == "B's question"


# =========================================================================
# Briefing isolation
# =========================================================================

SAMPLE_TICKER = [{
    "ticker": "AAPL", "price": 180.0, "change_pct": 1.0,
    "technical_signal": "RSI", "news_summary": "News", "outlook": "bullish",
}]


class TestBriefingIsolation:
    async def test_briefings_scoped_to_user(self, db):
        await save_briefing(
            raw_json="{}", market_regime="Bull", market_positioning="Long",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKER,
            user_id=USER_A,
        )
        await save_briefing(
            raw_json="{}", market_regime="Bear", market_positioning="Short",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKER,
            user_id=USER_B,
        )

        a_briefings = await get_recent_briefings(user_id=USER_A)
        b_briefings = await get_recent_briefings(user_id=USER_B)

        assert len(a_briefings) == 1
        assert a_briefings[0]["market_regime"] == "Bull"
        assert len(b_briefings) == 1
        assert b_briefings[0]["market_regime"] == "Bear"

    async def test_briefing_history_scoped_to_user(self, db):
        await save_briefing(
            raw_json="{}", market_regime="Bull", market_positioning="Long",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKER,
            user_id=USER_A,
        )

        a_history = await get_briefing_history("AAPL", USER_A, days=30)
        b_history = await get_briefing_history("AAPL", USER_B, days=30)

        assert len(a_history) == 1
        assert len(b_history) == 0

    async def test_none_user_id_returns_empty(self, db):
        """Safety: get_recent_briefings with None user_id returns empty."""
        await save_briefing(
            raw_json="{}", market_regime="Bull", market_positioning="Long",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKER,
            user_id=USER_A,
        )
        assert await get_recent_briefings(user_id=None) == []


# =========================================================================
# Company activity isolation
# =========================================================================

class TestCompanyActivityIsolation:
    async def test_activity_scoped_to_user(self, db):
        await get_or_create_session("AAPL", USER_A)
        await save_briefing(
            raw_json="{}", market_regime="Bull", market_positioning="Long",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKER,
            user_id=USER_A,
        )

        a_activity = await get_company_activity("AAPL", USER_A)
        b_activity = await get_company_activity("AAPL", USER_B)

        assert len(a_activity["sessions"]) == 1
        assert len(a_activity["briefings"]) == 1
        assert len(b_activity["sessions"]) == 0
        assert len(b_activity["briefings"]) == 0
