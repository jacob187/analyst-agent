"""Tests for api/enrichment.py — lazy company enrichment."""

from unittest.mock import MagicMock, patch
import pytest
from api.db import init_db, ensure_company, get_company, update_company
from api.enrichment import enrich_company


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """Isolated SQLite DB for each test."""
    import api.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "_db", None)
    await init_db()
    yield
    await db_mod.close_db()


class TestEnrichCompany:
    async def test_enriches_from_yfinance(self, db):
        await ensure_company("AAPL")

        mock_retriever = MagicMock()
        mock_retriever.ticker.info = {
            "shortName": "Apple Inc.",
            "sector": "Technology",
        }

        with patch(
            "agents.technical_workflow.get_stock_data.YahooFinanceDataRetrieval",
            return_value=mock_retriever,
        ):
            result = await enrich_company("AAPL")

        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"

        # Verify persisted to DB
        company = await get_company("AAPL")
        assert company["name"] == "Apple Inc."

    async def test_skips_if_already_enriched(self, db):
        await ensure_company("AAPL")
        await update_company("AAPL", name="Apple Inc.", sector="Technology")

        with patch("agents.technical_workflow.get_stock_data.YahooFinanceDataRetrieval") as mock_cls:
            result = await enrich_company("AAPL")
            # Should NOT call yfinance — already enriched
            mock_cls.assert_not_called()

        assert result["name"] == "Apple Inc."

    async def test_returns_none_for_unknown_ticker(self, db):
        result = await enrich_company("ZZZZ")
        assert result is None

    async def test_handles_yfinance_error(self, db):
        await ensure_company("AAPL")

        with patch(
            "agents.technical_workflow.get_stock_data.YahooFinanceDataRetrieval",
            side_effect=Exception("yfinance down"),
        ):
            result = await enrich_company("AAPL")

        # Should not crash — returns company with name=None
        assert result is not None
        assert result["name"] is None
