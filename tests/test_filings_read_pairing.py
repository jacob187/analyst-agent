"""The company page keys each 10-K/10-Q section by the filing its input was
actually read from, not by metadata read once before extraction: a
concurrent freshness refresh can move the shared retriever to a newer filing
mid-request, and an analysis saved under the wrong accession would be served
as that filing's for good.

Real ``SECDataRetrieval`` over the in-memory EDGAR in tests/sec_fakes.py.
"""

import threading
from datetime import date

import pytest

from agents.sec_workflow import analysis_cache
from api.routes.company import _fetch_filing_data, _paired_read, _plan_sections
from tests.sec_fakes import FakeEdgar, FakeFiling, FakeReport, eightk, tenk, tenq

TICKER = "PAIR"
K1, K2, K3 = "0000320193-25-000101", "0000320193-26-000202", "0000320193-26-000303"
Q1 = "0000320193-26-000111"
E1 = "0000320193-26-000121"


@pytest.fixture
def edgar(monkeypatch):
    analysis_cache.clear_caches()
    fake = FakeEdgar()
    fake.publish(tenk(K1, "2025-11-01", "OLD-RISK", mda="OLD-MDA"))
    fake.publish(tenq(Q1, "2026-02-01", "Quarterly risk factors."))
    fake.publish(eightk(E1, "2026-01-30"))
    fake.install(monkeypatch)
    yield fake
    analysis_cache.clear_caches()


def test_sections_read_after_a_refresh_are_keyed_by_the_new_filing(edgar):
    retriever = analysis_cache.get_retriever(TICKER)
    old_report = edgar.filings["10-K"][0].report
    gate, entered = threading.Event(), threading.Event()
    old_report.gates["Item 1A"] = gate
    old_report.entered["Item 1A"] = entered
    data: dict = {}
    worker = threading.Thread(target=lambda: data.update(_fetch_filing_data(TICKER, True)))
    worker.start()
    try:
        # The page is reading the old 10-K's risk factors when a chat lookup
        # finds a newer 10-K and moves the shared retriever to it.
        assert entered.wait(5)
        edgar.publish(tenk(K2, "2026-11-01", "NEW-RISK", mda="NEW-MDA"))
        retriever.refresh_latest_filings(("10-K",))
    finally:
        gate.set()
        worker.join(5)
    assert not worker.is_alive()

    plan = {p["key"]: p for p in _plan_sections(data)}
    assert data["tenk"]["metadata"]["accession"] == K1
    assert plan["mda_10k"]["input"]["text"] == "NEW-MDA"
    assert plan["mda_10k"]["accession"] == K2
    assert plan["balance"]["accession"] == K2
    assert plan["risk_10k"]["input"]["text"] == "OLD-RISK"
    assert plan["risk_10k"]["accession"] == K1
    assert plan["risk_10q"]["accession"] == Q1
    assert plan["mda_10k"]["filing_date"] == "2026-11-01"
    assert plan["risk_10k"]["filing_date"] == "2025-11-01"


def test_paired_read_retries_a_read_that_straddled_a_refresh(edgar):
    retriever = analysis_cache.get_retriever(TICKER)
    retriever.get_tenk()
    edgar.publish(tenk(K2, "2026-11-01", "NEW-RISK"))
    swapped: list[bool] = []

    def read():
        if not swapped:
            swapped.append(True)
            retriever.refresh_latest_filings(("10-K",))
        return retriever.get_tenk()

    value, meta = _paired_read(retriever, "10-K", read)
    assert value is edgar.filings["10-K"][0].report
    assert meta["accession"] == K2


def test_paired_read_drops_a_read_that_keeps_straddling_refreshes(edgar):
    retriever = analysis_cache.get_retriever(TICKER)
    retriever.get_tenk()
    newer = [tenk(K3, "2026-12-01", "R3"), tenk(K2, "2026-11-01", "R2")]

    def read():
        if newer:
            edgar.publish(newer.pop())
            retriever.refresh_latest_filings(("10-K",))
        return "statement"

    assert _paired_read(retriever, "10-K", read) == (None, {})


def test_paired_read_of_a_missing_form_is_empty_not_an_error(monkeypatch):
    analysis_cache.clear_caches()
    FakeEdgar().install(monkeypatch)
    retriever = analysis_cache.get_retriever(TICKER)
    assert _paired_read(retriever, "10-Q", lambda: "statement") == (None, {})
    analysis_cache.clear_caches()


def test_missing_balance_sheet_does_not_drop_the_10k(monkeypatch):
    """No XBRL balance sheet: only that section is unusable (cache_only);
    the rest of the 10-K is still analysed."""
    analysis_cache.clear_caches()
    fake = FakeEdgar()
    report = FakeReport({"Item 1A": "Risk text.", "Item 7": "MD&A text."}, balance_sheet=None)
    fake.publish(FakeFiling("10-K", K1, date(2025, 11, 1), report))
    fake.install(monkeypatch)
    try:
        data = _fetch_filing_data(TICKER, True)
    finally:
        analysis_cache.clear_caches()

    plan = {p["key"]: p for p in _plan_sections(data)}
    assert data["tenk"]["metadata"]["form_type"] == "10-K"
    assert plan["balance"]["cache_only"] is True
    assert plan["risk_10k"]["cache_only"] is False
    assert plan["mda_10k"]["input"]["text"] == "MD&A text."
