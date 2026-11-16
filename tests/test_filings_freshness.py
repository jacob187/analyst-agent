"""The company filings loader (``api/routes/company.py:_load_filing_data``)
syncs the shared retriever with EDGAR before reading filing metadata: one
submissions request for every form, 20-F filers included, and a failed
lookup is logged rather than failing the page.

The real retriever runs over the in-memory EDGAR in tests/sec_fakes.py.
"""

import logging

import pytest

from api.routes.company import _load_filing_data
from tests.sec_fakes import FakeEdgar, eightk, tenk, tenq, twentyf

TICKER = "TSTC"
K1, K2 = "0000320193-25-000101", "0000320193-26-000202"
Q1 = "0000320193-26-000111"
E1 = "0000320193-26-000121"
F1, F2 = "0000320193-25-000301", "0000320193-26-000302"


@pytest.fixture(autouse=True)
def _default_ttl(monkeypatch):
    monkeypatch.delenv("FILING_FRESHNESS_TTL_SECONDS", raising=False)


@pytest.fixture
def edgar(monkeypatch):
    fake = FakeEdgar()
    fake.publish(tenk(K1, "2025-11-01", "K1-RISK"))
    fake.publish(tenq(Q1, "2026-02-01", "Q1-RISK"))
    fake.publish(eightk(E1, "2026-01-30"))
    return fake.install(monkeypatch)


@pytest.fixture
def foreign_filer(monkeypatch):
    fake = FakeEdgar()
    fake.publish(twentyf(F1, "2025-04-01", "F1-RISK"))
    return fake.install(monkeypatch)


async def test_cold_load_fetches_the_submissions_once(edgar):
    data = await _load_filing_data(TICKER, fetch_raw=False)

    assert data["tenk"]["metadata"]["accession"] == K1
    assert data["tenq"]["metadata"]["accession"] == Q1
    assert data["eightk"]["metadata"]["accession"] == E1
    assert edgar.companies == 1


async def test_reload_within_ttl_asks_edgar_nothing(edgar):
    await _load_filing_data(TICKER, fetch_raw=False)
    built, requests = edgar.companies, edgar.requests

    await _load_filing_data(TICKER, fetch_raw=False)

    assert (edgar.companies, edgar.requests) == (built, requests)


async def test_foreign_filer_is_served_its_20f(foreign_filer):
    data = await _load_filing_data(TICKER, fetch_raw=True)

    assert data["tenk"]["metadata"]["form_type"] == "20-F"
    assert data["tenk"]["metadata"]["accession"] == F1
    assert data["tenk"]["risk_raw"]["text"] == "F1-RISK"


async def test_foreign_filers_new_20f_is_picked_up_after_the_ttl(foreign_filer, monkeypatch):
    await _load_filing_data(TICKER, fetch_raw=True)
    foreign_filer.publish(twentyf(F2, "2026-04-01", "F2-RISK"))

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "3600")
    within_ttl = await _load_filing_data(TICKER, fetch_raw=True)
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    after_ttl = await _load_filing_data(TICKER, fetch_raw=True)

    assert within_ttl["tenk"]["metadata"]["accession"] == F1
    assert after_ttl["tenk"]["metadata"]["accession"] == F2
    assert after_ttl["tenk"]["risk_raw"]["text"] == "F2-RISK"


async def test_failed_lookup_after_ttl_serves_current_filings_and_logs(edgar, monkeypatch, caplog):
    await _load_filing_data(TICKER, fetch_raw=False)
    edgar.publish(tenk(K2, "2026-11-01", "K2-RISK"))
    edgar.outage = RuntimeError("EDGAR 429")
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")

    with caplog.at_level(logging.WARNING):
        data = await _load_filing_data(TICKER, fetch_raw=False)

    assert data["tenk"]["metadata"]["accession"] == K1
    assert any("EDGAR 429" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


async def test_cold_failed_lookup_is_logged_before_the_load_fails(edgar, caplog):
    edgar.outage = RuntimeError("EDGAR 503")

    with caplog.at_level(logging.WARNING, logger="analyst.filings"):
        with pytest.raises(RuntimeError, match="EDGAR 503"):
            await _load_filing_data(TICKER, fetch_raw=False)

    assert any(
        "lookup failed" in r.getMessage() and "EDGAR 503" in r.getMessage()
        for r in caplog.records if r.name == "analyst.filings"
    )
