"""Section extraction on one ``SECDataRetrieval`` runs once per
(form, accession, section) and is shared by concurrent callers.

The real retriever runs over the in-memory EDGAR in tests/sec_fakes.py; the
fake report counts parses and per-section extractions, and real worker
threads (``asyncio.to_thread`` / a thread pool) hit it concurrently.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from tests.sec_fakes import FakeEdgar, eightk, tenk, tenq

TICKER = "TSTC"
K1 = "0000320193-25-000101"
Q1, Q2 = "0000320193-26-000111", "0000320193-26-000222"
E1 = "0000320193-26-000121"
Q1_RISK = "Q1-RISK: supply chain concentration."
Q2_RISK = "Q2-RISK: new export controls on advanced chips."
Q1_MDA = "Q1-MDA: services revenue grew 12%."
N = 8


@pytest.fixture
def edgar(monkeypatch):
    fake = FakeEdgar()
    fake.publish(tenk(K1, "2025-11-01", "K1-RISK"))
    fake.publish(tenq(Q1, "2026-02-01", Q1_RISK, mda=Q1_MDA))
    fake.publish(eightk(E1, "2026-01-30", earnings=True, item_text="Item 2.02 results"))
    return fake.install(monkeypatch)


@pytest.fixture
def retriever(edgar):
    return SECDataRetrieval(TICKER)


def _latest(edgar: FakeEdgar, form: str):
    return edgar.filings[form][0]


def _in_threads(fn, *calls):
    """Run ``fn(*args)`` for each args tuple on its own thread, all released
    together; return the results in order."""
    barrier = threading.Barrier(len(calls))

    def run(args):
        barrier.wait()
        return fn(*args)

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(run, calls))


# ── AC1: concurrent identical requests extract once ──────────────────────────


async def test_concurrent_same_section_extracts_once(edgar, retriever):
    filing = _latest(edgar, "10-Q")
    filing.report.read_delay = 0.05  # first extraction still running as others arrive

    results = await asyncio.gather(
        *(asyncio.to_thread(retriever.get_risk_factors_raw, "10-Q") for _ in range(N))
    )

    assert all(r["found"] and r["text"] == Q1_RISK for r in results)
    assert all(r["metadata"]["accession"] == Q1 for r in results)
    assert filing.report.reads["Item 1A"] == 1
    assert filing.obj_calls == 1
    assert filing.report.parses == 1


def test_concurrent_different_sections_share_one_filing_object_and_parse(edgar, retriever):
    """The logged bug: risk factors, MD&A and market risk on a cold retriever
    each built their own report object and re-ran section detection."""
    filing = _latest(edgar, "10-Q")
    filing.obj_delay = 0.05
    filing.report.parse_delay = 0.05

    results = _in_threads(retriever.get_section, *[("10-Q", item) for item in ("1A", "2", "3")] * 3)

    assert [r["text"] for r in results[:2]] == [Q1_RISK, Q1_MDA]
    assert filing.obj_calls == 1
    assert filing.report.parses == 1
    assert filing.report.reads["Item 1A"] == 1
    assert filing.report.reads["Item 2"] == 1


def test_concurrent_8k_item_extracts_once(edgar, retriever):
    filing = _latest(edgar, "8-K")
    filing.report.read_delay = 0.05

    results = _in_threads(retriever.get_8k_item, *[("2.02",)] * N)

    assert all(r["found"] and r["text"] == "Item 2.02 results" for r in results)
    assert filing.report.reads["2.02"] == 1
    assert filing.obj_calls == 1


def test_repeat_calls_reuse_the_extraction(edgar, retriever):
    report = _latest(edgar, "10-Q").report
    for _ in range(3):
        assert retriever.get_mda_raw("10-Q")["text"] == Q1_MDA
    assert report.reads["Item 2"] == 1


def test_mutating_a_result_does_not_change_the_memo(edgar, retriever):
    first = retriever.get_risk_factors_raw("10-Q")
    first["text"] = "edited"
    first["metadata"]["accession"] = "edited"

    again = retriever.get_risk_factors_raw("10-Q")

    assert again["text"] == Q1_RISK
    assert again["metadata"]["accession"] == Q1


# ── AC2: different sections extract independently ───────────────────────────


def test_slow_section_does_not_block_another_section(edgar, retriever):
    report = _latest(edgar, "10-Q").report
    release_a = threading.Event()
    report.gates["Item 1A"] = release_a
    report.entered["Item 1A"] = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow_a = pool.submit(retriever.get_risk_factors_raw, "10-Q")
        assert report.entered["Item 1A"].wait(timeout=5), "section A never started extracting"

        fast_b = pool.submit(retriever.get_mda_raw, "10-Q")
        b = fast_b.result(timeout=5)  # times out if B waits behind A

        assert b["found"] and b["text"] == Q1_MDA
        assert not slow_a.done()
        release_a.set()
        a = slow_a.result(timeout=5)

    assert a["found"] and a["text"] == Q1_RISK
    assert report.reads["Item 1A"] == 1
    assert report.reads["Item 2"] == 1


# ── AC3: a newer filing re-extracts ──────────────────────────────────────────


def test_refresh_to_new_accession_reextracts_from_new_filing(edgar, retriever):
    old = _latest(edgar, "10-Q")
    assert retriever.get_risk_factors_raw("10-Q")["text"] == Q1_RISK

    edgar.publish(tenq(Q2, "2026-05-01", Q2_RISK))
    meta = retriever.refresh_latest_filing("10-Q")
    result = retriever.get_risk_factors_raw("10-Q")

    new = _latest(edgar, "10-Q")
    assert meta.accession == Q2
    assert result["found"] and result["text"] == Q2_RISK
    assert result["metadata"]["accession"] == Q2
    assert new.report.reads["Item 1A"] == 1
    assert old.report.reads["Item 1A"] == 1


def test_refresh_with_same_accession_keeps_the_memo(edgar, retriever):
    report = _latest(edgar, "10-Q").report
    retriever.get_risk_factors_raw("10-Q")

    retriever.refresh_latest_filing("10-Q")
    result = retriever.get_risk_factors_raw("10-Q")

    assert result["text"] == Q1_RISK
    assert report.reads["Item 1A"] == 1


# ── AC4: failures are not memoized ───────────────────────────────────────────


def test_transient_failure_is_retried_on_next_call(edgar, retriever):
    report = _latest(edgar, "10-Q").report
    report.failures["Item 1A"] = 1

    failed = retriever.get_risk_factors_raw("10-Q")
    retried = retriever.get_risk_factors_raw("10-Q")
    again = retriever.get_risk_factors_raw("10-Q")

    assert failed["found"] is False and "Error extracting" in failed["text"]
    assert retried["found"] is True and retried["text"] == Q1_RISK
    assert again["text"] == Q1_RISK
    assert report.reads["Item 1A"] == 2


def test_not_found_result_is_retried_on_next_call(edgar, retriever):
    """Getters report a miss as a ``found: False`` result, not an exception,
    and a miss can't be told apart from a transient failure, so it is not
    memoized."""
    report = _latest(edgar, "10-Q").report
    report.blanks["Item 2"] = 1

    missing = retriever.get_mda_raw("10-Q")
    retried = retriever.get_mda_raw("10-Q")
    again = retriever.get_mda_raw("10-Q")

    assert missing["found"] is False
    assert retried["found"] is True and retried["text"] == Q1_MDA
    assert again["text"] == Q1_MDA
    assert report.reads["Item 2"] == 2
