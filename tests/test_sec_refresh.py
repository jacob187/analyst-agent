"""``SECDataRetrieval`` filing lookup: one ``FilingMetadata`` builder shared by
the lazy fetchers and the freshness refresh, and one submissions request
(``Company`` build) for a refresh of several forms.

The real retriever runs over the in-memory EDGAR in tests/sec_fakes.py.
"""

import pytest

from agents.sec_workflow.get_SEC_data import FilingMetadata, SECDataRetrieval
from tests.sec_fakes import CIK, COMPANY_NAME, FakeEdgar, eightk, tenk, tenq

TICKER = "TSTC"
K1, K2 = "0000320193-25-000101", "0000320193-26-000202"
Q1 = "0000320193-26-000111"
E1, E2 = "0000320193-26-000121", "0000320193-26-000222"


@pytest.fixture
def edgar(monkeypatch):
    fake = FakeEdgar()
    fake.publish(tenk(K1, "2025-11-01", "K1-RISK"))
    fake.publish(tenq(Q1, "2026-02-01", "Q1-RISK"))
    fake.publish(eightk(E1, "2026-01-30"))
    return fake.install(monkeypatch)


def test_from_filing_maps_every_field(edgar):
    filing = edgar.filings["10-K"][0]
    meta = FilingMetadata.from_filing(filing)
    assert meta.to_dict() == {
        "form": "10-K",
        "cik": str(CIK),
        "accession": K1,
        "filing_date": "2025-11-01",
        "period_of_report": filing.period_of_report,
        "company_name": COMPANY_NAME,
    }


def test_lazy_fetch_and_refresh_build_the_same_metadata(edgar):
    lazy = SECDataRetrieval(TICKER)
    lazy.get_tenk_filing()
    refreshed = SECDataRetrieval(TICKER).refresh_latest_filing("10-K")
    assert refreshed.to_dict() == lazy._tenk_metadata.to_dict()


def test_refreshing_several_forms_builds_one_company(edgar):
    retriever = SECDataRetrieval(TICKER)
    built = edgar.companies

    metas = retriever.refresh_latest_filings(("10-K", "10-Q", "8-K", "20-F"))

    assert edgar.companies == built + 1
    assert metas["10-K"].accession == K1
    assert metas["10-Q"].accession == Q1
    assert metas["8-K"].accession == E1
    assert metas["20-F"] is None


def test_refresh_swaps_each_form_to_its_new_filing(edgar):
    retriever = SECDataRetrieval(TICKER)
    assert retriever.get_risk_factors_raw("10-K")["text"] == "K1-RISK"
    edgar.publish(tenk(K2, "2026-11-01", "K2-RISK"))
    edgar.publish(eightk(E2, "2026-11-10", earnings=False))

    metas = retriever.refresh_latest_filings(("10-K", "8-K"))

    assert (metas["10-K"].accession, metas["8-K"].accession) == (K2, E2)
    assert retriever.get_risk_factors_raw("10-K")["text"] == "K2-RISK"
    assert retriever.get_8k_overview()["metadata"]["accession"] == E2
