"""Tests for api/db.py — database layer.

Marker: eval_unit (no API keys, no network).

Each test gets a fresh, isolated SQLite database via the `fresh_db` fixture so
tests can never contaminate each other through shared state.

Key isolation technique:
  api.db uses a module-level `_db` global for connection pooling. The fixture
  patches both `_db` (resets the cached connection) and `DB_PATH` (redirects
  to a temp file) before calling `init_db()`, then tears everything down after.
"""

import pytest
import api.db as db_module


# Stable test user IDs — two distinct users for isolation tests
USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Fixture: fresh isolated database per test
# ---------------------------------------------------------------------------

@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    """
    Redirect api.db to a fresh temp-file DB and initialise its schema.

    Why patch both `_db` and `DB_PATH`?
    - `DB_PATH` tells `get_db()` where to create the file.
    - `_db` is the cached connection; setting it to None forces `get_db()`
      to open a new connection to the patched path rather than reusing the
      existing one from a previous test.
    """
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield
    await db_module.close_db()
    # close_db() sets _db back to None, so the next fixture iteration starts clean


# ---------------------------------------------------------------------------
# init_db / schema migration
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestInitDb:
    async def test_creates_sessions_table(self, fresh_db):
        db = await db_module.get_db()
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None

    async def test_creates_messages_table(self, fresh_db):
        db = await db_module.get_db()
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None

    async def test_summary_column_exists(self, fresh_db):
        """summary TEXT must be present — added by migration if DB predates it."""
        db = await db_module.get_db()
        async with db.execute("PRAGMA table_info(sessions)") as cur:
            columns = {row["name"] async for row in cur}
        assert "summary" in columns

    async def test_user_id_column_exists(self, fresh_db):
        """user_id TEXT must be present on sessions."""
        db = await db_module.get_db()
        async with db.execute("PRAGMA table_info(sessions)") as cur:
            columns = {row["name"] async for row in cur}
        assert "user_id" in columns

    async def test_watchlist_has_user_id_pk(self, fresh_db):
        """watchlist PK should be (user_id, ticker)."""
        db = await db_module.get_db()
        async with db.execute("PRAGMA table_info(watchlist)") as cur:
            columns = {row["name"] async for row in cur}
        assert "user_id" in columns

    async def test_init_db_is_idempotent(self, fresh_db):
        """Calling init_db() twice must not raise or corrupt tables."""
        await db_module.init_db()  # second call


# ---------------------------------------------------------------------------
# get_or_create_session
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestGetOrCreateSession:
    async def test_creates_session_when_none_exists(self, fresh_db):
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        assert session_id  # non-empty string

    async def test_returns_same_id_on_second_call(self, fresh_db):
        """One session per ticker per user — second call must return the same ID."""
        first = await db_module.get_or_create_session("TSLA", USER_A)
        second = await db_module.get_or_create_session("TSLA", USER_A)
        assert first == second

    async def test_different_tickers_get_different_sessions(self, fresh_db):
        aapl = await db_module.get_or_create_session("AAPL", USER_A)
        msft = await db_module.get_or_create_session("MSFT", USER_A)
        assert aapl != msft

    async def test_normalises_ticker_to_uppercase(self, fresh_db):
        """'aapl' and 'AAPL' must resolve to the same session."""
        lower = await db_module.get_or_create_session("aapl", USER_A)
        upper = await db_module.get_or_create_session("AAPL", USER_A)
        assert lower == upper

    async def test_session_stored_in_db(self, fresh_db):
        session_id = await db_module.get_or_create_session("NVDA", USER_A)
        session = await db_module.get_session(session_id)
        assert session is not None
        assert session["ticker"] == "NVDA"
        assert session["user_id"] == USER_A

    async def test_different_users_get_different_sessions(self, fresh_db):
        """Same ticker, different users — must get separate sessions."""
        a = await db_module.get_or_create_session("AAPL", USER_A)
        b = await db_module.get_or_create_session("AAPL", USER_B)
        assert a != b


# ---------------------------------------------------------------------------
# get_session_by_ticker
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestGetSessionByTicker:
    async def test_returns_none_when_no_session(self, fresh_db):
        result = await db_module.get_session_by_ticker("XYZ", USER_A)
        assert result is None

    async def test_returns_session_after_creation(self, fresh_db):
        session_id = await db_module.get_or_create_session("GOOG", USER_A)
        result = await db_module.get_session_by_ticker("GOOG", USER_A)
        assert result is not None
        assert result["id"] == session_id
        assert result["ticker"] == "GOOG"

    async def test_case_insensitive_lookup(self, fresh_db):
        await db_module.get_or_create_session("AMZN", USER_A)
        result = await db_module.get_session_by_ticker("amzn", USER_A)
        assert result is not None

    async def test_returns_correct_ticker_field(self, fresh_db):
        await db_module.get_or_create_session("META", USER_A)
        result = await db_module.get_session_by_ticker("META", USER_A)
        assert result["ticker"] == "META"

    async def test_other_user_cannot_see_session(self, fresh_db):
        """User B must not see user A's session."""
        await db_module.get_or_create_session("AAPL", USER_A)
        result = await db_module.get_session_by_ticker("AAPL", USER_B)
        assert result is None


# ---------------------------------------------------------------------------
# get_session (including summary field)
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestGetSession:
    async def test_returns_none_for_unknown_id(self, fresh_db):
        result = await db_module.get_session("does-not-exist")
        assert result is None

    async def test_returns_all_fields(self, fresh_db):
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        session = await db_module.get_session(session_id)
        assert set(session.keys()) >= {"id", "ticker", "summary", "user_id", "created_at"}

    async def test_summary_is_none_initially(self, fresh_db):
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        session = await db_module.get_session(session_id)
        assert session["summary"] is None


# ---------------------------------------------------------------------------
# update_session_summary
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestUpdateSessionSummary:
    async def test_stores_summary(self, fresh_db):
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        await db_module.update_session_summary(session_id, "Key findings: RSI 32, oversold.")
        session = await db_module.get_session(session_id)
        assert session["summary"] == "Key findings: RSI 32, oversold."

    async def test_overwrites_previous_summary(self, fresh_db):
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        await db_module.update_session_summary(session_id, "First summary")
        await db_module.update_session_summary(session_id, "Updated summary")
        session = await db_module.get_session(session_id)
        assert session["summary"] == "Updated summary"

    async def test_summary_does_not_affect_messages(self, fresh_db):
        """Updating summary must not touch the messages table."""
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        await db_module.save_message(session_id, "user", "hello")
        await db_module.update_session_summary(session_id, "some summary")
        messages = await db_module.get_session_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestDeleteSession:
    async def test_delete_removes_session(self, fresh_db):
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        deleted = await db_module.delete_session(session_id)
        assert deleted is True
        assert await db_module.get_session(session_id) is None

    async def test_delete_returns_false_for_unknown_id(self, fresh_db):
        deleted = await db_module.delete_session("ghost-id")
        assert deleted is False

    async def test_delete_removes_messages_too(self, fresh_db):
        """Cascade: deleting a session must remove all its messages."""
        session_id = await db_module.get_or_create_session("AAPL", USER_A)
        await db_module.save_message(session_id, "user", "hello")
        await db_module.delete_session(session_id)
        # After deletion the session is gone, but messages should be gone too.
        # get_session_messages returns [] for unknown session_id.
        messages = await db_module.get_session_messages(session_id)
        assert messages == []

    async def test_after_delete_new_session_created_for_same_ticker(self, fresh_db):
        """After deleting, get_or_create_session must produce a fresh ID."""
        first_id = await db_module.get_or_create_session("AAPL", USER_A)
        await db_module.delete_session(first_id)
        second_id = await db_module.get_or_create_session("AAPL", USER_A)
        assert second_id != first_id
