"""``SECDataRetrieval`` filing lookup: one ``FilingMetadata`` builder shared by
the lazy fetchers and the freshness refresh, and one submissions request
(``Company`` build) for a refresh of several forms.

The real retriever runs over the in-memory EDGAR in tests/sec_fakes.py.
"""

import pytest

from agents.sec_workflow.get_SEC_data import FilingMetadata, SECDataRetrieval
from tests.sec_fakes import CIK, COMPANY_NAME, FakeEdgar, eightk, tenk, tenq, twentyf

TICKER = "TSTC"
K1, K2 = "0000320193-25-000101", "0000320193-26-000202"
Q1 = "0000320193-26-000111"
E1, E2 = "0000320193-26-000121", "0000320193-26-000222"
F1 = "0000320193-26-000301"


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


# ── A refresh reads EDGAR's recent submissions page, not the whole history ──

FORMS = ("10-K", "10-Q", "8-K", "20-F")


def test_repeat_refreshes_download_no_older_pages(edgar):
    # The first refresh looks through the full history once (this filer has
    # no 20-F); later refreshes, each with a fresh Company, read only the
    # recent page, and still see a newly published 8-K.
    edgar.older_pages = 3
    retriever = SECDataRetrieval(TICKER)
    retriever.refresh_latest_filings(FORMS, reuse_company=True)
    first = edgar.page_downloads
    assert first == 3

    edgar.publish(eightk(E2, "2026-11-10", earnings=False))
    for _ in range(3):
        metas = retriever.refresh_latest_filings(FORMS)

    assert metas["8-K"].accession == E2
    assert metas["20-F"] is None
    assert edgar.page_downloads == first


def test_filing_beyond_the_recent_page_is_kept_without_reloading(edgar):
    # A heavy filer: its 10-K has rolled off the recent page (only the two
    # newest filings are on it). The full history is read once to find it;
    # later refreshes keep it until a newer 10-K lands on the recent page.
    edgar.recent_size = 2
    retriever = SECDataRetrieval(TICKER)

    assert retriever.refresh_latest_filings(FORMS)["10-K"].accession == K1
    downloads = edgar.page_downloads
    assert retriever.refresh_latest_filings(FORMS)["10-K"].accession == K1
    assert edgar.page_downloads == downloads

    edgar.publish(tenk(K2, "2026-11-01", "K2-RISK"))
    metas = retriever.refresh_latest_filings(FORMS)
    assert metas["10-K"].accession == K2
    assert retriever.get_risk_factors_raw("10-K")["text"] == "K2-RISK"
    assert edgar.page_downloads == downloads


def test_form_first_filed_after_the_history_lookup_is_found(edgar):
    retriever = SECDataRetrieval(TICKER)
    assert retriever.refresh_latest_filings(FORMS)["20-F"] is None

    edgar.publish(twentyf(F1, "2026-11-12", "F1-RISK"))

    assert retriever.refresh_latest_filings(FORMS)["20-F"].accession == F1


def test_lazy_fetch_reads_only_the_recent_page(edgar):
    retriever = SECDataRetrieval(TICKER)
    retriever.get_eightk_filing()
    retriever.get_tenk_filing()
    assert edgar.page_downloads == 0
