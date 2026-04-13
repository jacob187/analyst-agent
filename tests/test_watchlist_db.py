"""Tests for watchlist database operations."""

import pytest
from api.db import (
    init_db,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
)

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
async def watchlist_db(tmp_path, monkeypatch):
    """Isolated DB for watchlist tests."""
    import api.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "_db", None)
    await init_db()
    yield
    await db_mod.close_db()


class TestGetWatchlist:
    async def test_empty(self, watchlist_db):
        result = await get_watchlist(USER_A)
        assert result == []

    async def test_returns_added(self, watchlist_db):
        await add_to_watchlist("AAPL", USER_A)
        await add_to_watchlist("MSFT", USER_A)
        result = await get_watchlist(USER_A)
        tickers = [r["ticker"] for r in result]
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestAddToWatchlist:
    async def test_add(self, watchlist_db):
        assert await add_to_watchlist("AAPL", USER_A) is True
        result = await get_watchlist(USER_A)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    async def test_duplicate_returns_false(self, watchlist_db):
        await add_to_watchlist("AAPL", USER_A)
        assert await add_to_watchlist("AAPL", USER_A) is False

    async def test_uppercase(self, watchlist_db):
        await add_to_watchlist("aapl", USER_A)
        result = await get_watchlist(USER_A)
        assert result[0]["ticker"] == "AAPL"

    async def test_limit_10(self, watchlist_db):
        for i in range(10):
            await add_to_watchlist(f"T{i}", USER_A)
        assert await add_to_watchlist("EXTRA", USER_A) is False

    async def test_add_after_remove(self, watchlist_db):
        await add_to_watchlist("AAPL", USER_A)
        await remove_from_watchlist("AAPL", USER_A)
        assert await add_to_watchlist("AAPL", USER_A) is True

    async def test_same_ticker_different_users(self, watchlist_db):
        """Two users can both add the same ticker."""
        assert await add_to_watchlist("AAPL", USER_A) is True
        assert await add_to_watchlist("AAPL", USER_B) is True

    async def test_limit_per_user(self, watchlist_db):
        """Each user gets their own 10-ticker limit."""
        for i in range(10):
            await add_to_watchlist(f"T{i}", USER_A)
        # User A is at limit, but User B should still be able to add
        assert await add_to_watchlist("AAPL", USER_B) is True


class TestRemoveFromWatchlist:
    async def test_remove_existing(self, watchlist_db):
        await add_to_watchlist("AAPL", USER_A)
        assert await remove_from_watchlist("AAPL", USER_A) is True
        assert await get_watchlist(USER_A) == []

    async def test_remove_nonexistent(self, watchlist_db):
        assert await remove_from_watchlist("XYZ", USER_A) is False

    async def test_remove_does_not_affect_other_user(self, watchlist_db):
        """Removing from user A's watchlist must not affect user B."""
        await add_to_watchlist("AAPL", USER_A)
        await add_to_watchlist("AAPL", USER_B)
        await remove_from_watchlist("AAPL", USER_A)
        assert await get_watchlist(USER_A) == []
        result_b = await get_watchlist(USER_B)
        assert len(result_b) == 1
        assert result_b[0]["ticker"] == "AAPL"
