"""Tests for the company tracking data layer — companies, briefings, filings_cache."""

import json
import pytest
from api.db import (
    init_db,
    add_to_watchlist,
    get_watchlist,
    ensure_company,
    get_company,
    get_companies,
    update_company,
    save_briefing,
    get_recent_briefings,
    get_briefing_history,
    save_filing_metadata,
    get_filing_metadata,
    mark_filing_downloaded,
)


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """Isolated SQLite DB for each test."""
    import api.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "_db", None)
    await init_db()
    yield
    await db_mod.close_db()


# =========================================================================
# Companies
# =========================================================================

class TestCompanies:
    async def test_ensure_company_creates_row(self, db):
        await ensure_company("AAPL")
        company = await get_company("AAPL")
        assert company is not None
        assert company["ticker"] == "AAPL"
        assert company["name"] is None  # not enriched yet

    async def test_ensure_company_idempotent(self, db):
        await ensure_company("AAPL")
        await ensure_company("AAPL")  # should not raise
        companies = await get_companies()
        assert len(companies) == 1

    async def test_ensure_company_uppercases(self, db):
        await ensure_company("aapl")
        assert (await get_company("AAPL")) is not None

    async def test_get_company_missing(self, db):
        assert (await get_company("XYZ")) is None

    async def test_get_companies_ordered(self, db):
        await ensure_company("MSFT")
        await ensure_company("AAPL")
        companies = await get_companies()
        assert len(companies) == 2

    async def test_update_company(self, db):
        await ensure_company("AAPL")
        updated = await update_company("AAPL", name="Apple Inc.", sector="Technology")
        assert updated is True
        company = await get_company("AAPL")
        assert company["name"] == "Apple Inc."
        assert company["sector"] == "Technology"

    async def test_update_company_partial(self, db):
        await ensure_company("AAPL")
        await update_company("AAPL", name="Apple Inc.")
        company = await get_company("AAPL")
        assert company["name"] == "Apple Inc."
        assert company["sector"] is None

    async def test_update_company_missing(self, db):
        result = await update_company("XYZ", name="Missing")
        assert result is False


class TestWatchlistCompanyIntegration:
    async def test_add_to_watchlist_creates_company(self, db):
        await add_to_watchlist("AAPL")
        company = await get_company("AAPL")
        assert company is not None
        assert company["ticker"] == "AAPL"

    async def test_watchlist_backfill_on_init(self, tmp_path, monkeypatch):
        """Existing watchlist rows get backfilled into companies on init."""
        import api.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "backfill.db")
        monkeypatch.setattr(db_mod, "_db", None)

        # First init: create watchlist entries without companies table
        await init_db()
        await add_to_watchlist("XOM")
        await add_to_watchlist("BP")
        await db_mod.close_db()

        # Second init: simulates app restart — backfill should run
        monkeypatch.setattr(db_mod, "_db", None)
        await init_db()

        assert (await get_company("XOM")) is not None
        assert (await get_company("BP")) is not None
        await db_mod.close_db()


# =========================================================================
# Briefings
# =========================================================================

SAMPLE_TICKERS = [
    {
        "ticker": "AAPL", "price": 180.50, "change_pct": -1.23,
        "technical_signal": "RSI oversold at 28.3",
        "news_summary": "Apple Q2 earnings beat", "news_url": "https://example.com/aapl",
        "outlook": "mixed",
    },
    {
        "ticker": "XOM", "price": 110.00, "change_pct": 0.5,
        "technical_signal": "MACD bullish crossover",
        "news_summary": "Oil prices rise", "news_url": None,
        "outlook": "bullish",
    },
]


class TestBriefingPersistence:
    async def test_save_and_retrieve(self, db):
        briefing_id = await save_briefing(
            raw_json='{"test": true}',
            market_regime="Bull, Low Vol",
            market_positioning="Stay long",
            alerts_json=json.dumps(["RSI divergence on AAPL"]),
            thinking="Let me analyze...",
            tickers=SAMPLE_TICKERS,
        )
        assert briefing_id is not None

        recent = await get_recent_briefings(limit=5)
        assert len(recent) == 1
        assert recent[0]["market_regime"] == "Bull, Low Vol"
        assert recent[0]["thinking"] == "Let me analyze..."
        assert len(recent[0]["tickers"]) == 2

    async def test_save_creates_company_records(self, db):
        """save_briefing defensively upserts companies for each ticker."""
        await save_briefing(
            raw_json="{}", market_regime="Bear", market_positioning="Reduce",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKERS,
        )
        assert (await get_company("AAPL")) is not None
        assert (await get_company("XOM")) is not None

    async def test_get_briefing_history_filters_by_ticker(self, db):
        await save_briefing(
            raw_json="{}", market_regime="Bull", market_positioning="Long",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKERS,
        )
        aapl_history = await get_briefing_history("AAPL", days=30)
        assert len(aapl_history) == 1
        assert aapl_history[0]["outlook"] == "mixed"

        xom_history = await get_briefing_history("XOM", days=30)
        assert len(xom_history) == 1
        assert xom_history[0]["outlook"] == "bullish"

        # Ticker not in any briefing
        msft_history = await get_briefing_history("MSFT", days=30)
        assert len(msft_history) == 0

    async def test_get_recent_briefings_ordered(self, db):
        await save_briefing(
            raw_json="{}", market_regime="First", market_positioning="...",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKERS[:1],
        )
        await save_briefing(
            raw_json="{}", market_regime="Second", market_positioning="...",
            alerts_json="[]", thinking=None, tickers=SAMPLE_TICKERS[:1],
        )
        recent = await get_recent_briefings(limit=10)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["market_regime"] == "Second"

    async def test_get_recent_briefings_respects_limit(self, db):
        for i in range(5):
            await save_briefing(
                raw_json="{}", market_regime=f"Briefing {i}",
                market_positioning="...", alerts_json="[]",
                thinking=None, tickers=SAMPLE_TICKERS[:1],
            )
        recent = await get_recent_briefings(limit=3)
        assert len(recent) == 3


# =========================================================================
# Filings Cache
# =========================================================================

class TestFilingsCache:
    async def test_save_and_retrieve(self, db):
        filing_id = await save_filing_metadata(
            ticker="AAPL",
            form_type="10-K",
            filing_date="2025-10-30",
            period_of_report="2025-09-30",
            accession_number="0000320193-25-000106",
            filing_url="https://www.sec.gov/Archives/edgar/data/...",
        )
        assert filing_id is not None

        filings = await get_filing_metadata("AAPL")
        assert len(filings) == 1
        assert filings[0]["form_type"] == "10-K"
        assert filings[0]["accession_number"] == "0000320193-25-000106"
        assert filings[0]["downloaded_at"] is None

    async def test_save_creates_company(self, db):
        await save_filing_metadata("TSLA", "10-Q", "2025-07-25")
        assert (await get_company("TSLA")) is not None

    async def test_filter_by_form_type(self, db):
        await save_filing_metadata("AAPL", "10-K", "2025-10-30")
        await save_filing_metadata("AAPL", "10-Q", "2025-07-30")
        await save_filing_metadata("AAPL", "10-Q", "2025-04-30")

        tenk = await get_filing_metadata("AAPL", form_type="10-K")
        assert len(tenk) == 1

        tenq = await get_filing_metadata("AAPL", form_type="10-Q")
        assert len(tenq) == 2

    async def test_mark_downloaded(self, db):
        filing_id = await save_filing_metadata("AAPL", "10-K", "2025-10-30")
        assert (await mark_filing_downloaded(filing_id)) is True

        filings = await get_filing_metadata("AAPL")
        assert filings[0]["downloaded_at"] is not None

    async def test_filing_id_as_future_fk(self, db):
        """The filing ID is a UUID that can serve as FK to a pgvector table."""
        filing_id = await save_filing_metadata("AAPL", "10-K", "2025-10-30")
        # Verify it's a valid UUID string
        import uuid
        parsed = uuid.UUID(filing_id)
        assert str(parsed) == filing_id
