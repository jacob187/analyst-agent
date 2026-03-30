"""Tests for watchlist database operations."""

import pytest
from api.db import (
    init_db,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
)


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
        result = await get_watchlist()
        assert result == []

    async def test_returns_added(self, watchlist_db):
        await add_to_watchlist("AAPL")
        await add_to_watchlist("MSFT")
        result = await get_watchlist()
        tickers = [r["ticker"] for r in result]
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestAddToWatchlist:
    async def test_add(self, watchlist_db):
        assert await add_to_watchlist("AAPL") is True
        result = await get_watchlist()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    async def test_duplicate_returns_false(self, watchlist_db):
        await add_to_watchlist("AAPL")
        assert await add_to_watchlist("AAPL") is False

    async def test_uppercase(self, watchlist_db):
        await add_to_watchlist("aapl")
        result = await get_watchlist()
        assert result[0]["ticker"] == "AAPL"

    async def test_limit_10(self, watchlist_db):
        for i in range(10):
            await add_to_watchlist(f"T{i}")
        assert await add_to_watchlist("EXTRA") is False

    async def test_add_after_remove(self, watchlist_db):
        await add_to_watchlist("AAPL")
        await remove_from_watchlist("AAPL")
        assert await add_to_watchlist("AAPL") is True


class TestRemoveFromWatchlist:
    async def test_remove_existing(self, watchlist_db):
        await add_to_watchlist("AAPL")
        assert await remove_from_watchlist("AAPL") is True
        assert await get_watchlist() == []

    async def test_remove_nonexistent(self, watchlist_db):
        assert await remove_from_watchlist("XYZ") is False
