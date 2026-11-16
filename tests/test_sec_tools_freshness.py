"""Chat SEC tools and filing freshness: every tool (raw text and LLM summary
alike) goes through the same latest-filing check, an analysis is saved
under the filing its text was read from, and a stored analysis is served
without downloading the filing.

Real temp SQLite, the real SECDataRetrieval over an in-memory EDGAR
(tests/sec_fakes.py), and a chat model that counts calls at the LLM boundary.
"""

import asyncio
import json

import pytest

import api.db as db_module
from agents.sec_workflow import analysis_cache
from agents.sec_workflow.analysis_cache import get_retriever, latest_filing
from agents.tools.sec_tools import create_sec_tools
from tests.sec_fakes import CountingChatModel, FakeEdgar, eightk, tenk, tenq

TICKER = "TSTC"
K1, K2 = "0000320193-25-000101", "0000320193-26-000202"
Q1 = "0000320193-26-000111"
E1, E2 = "0000320193-26-000121", "0000320193-26-000222"
K1_RISK = "K1-RISK: supply chain concentration and antitrust exposure."
K2_RISK = "K2-RISK: new export controls on advanced chips."
TENQ_RISK = "There have been no material changes to the risk factors previously disclosed."


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield
    await db_module.close_db()


@pytest.fixture
def edgar(monkeypatch):
    monkeypatch.delenv("FILING_FRESHNESS_TTL_SECONDS", raising=False)
    fake = FakeEdgar()
    fake.publish(tenk(K1, "2025-11-01", K1_RISK))
    fake.publish(tenq(Q1, "2026-02-01", TENQ_RISK, mda="Q1-MDA: services revenue grew 12%."))
    fake.publish(eightk(E1, "2026-01-30", earnings=True, item_text="Q1 results."))
    return fake.install(monkeypatch)


@pytest.fixture
def llm():
    return CountingChatModel()


async def _ask(llm, name: str, args="") -> str:
    tools = {t.name: t for t in create_sec_tools(TICKER, llm)}
    return await tools[name].ainvoke(args)


async def _row(form: str, accession: str, analysis_type: str):
    return await db_module.get_filing_analysis(TICKER, form, accession, analysis_type)


def _restart_process() -> None:
    analysis_cache.clear_caches()


# ── #1: raw tools go through the freshness check too ─────────────────────────


async def test_raw_risk_factors_read_the_new_10k_after_the_ttl(edgar, llm, monkeypatch):
    assert await _ask(llm, "get_raw_risk_factors") == K1_RISK
    edgar.publish(tenk(K2, "2026-11-01", K2_RISK))

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    assert await _ask(llm, "get_raw_risk_factors") == K2_RISK


async def test_8k_item_reads_the_new_8k_after_the_ttl(edgar, llm, monkeypatch):
    assert await _ask(llm, "get_8k_item", {"item": "2.02"}) == "Q1 results."
    edgar.publish(eightk(E2, "2026-11-10", earnings=False, item_text="CEO resigned."))

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    assert await _ask(llm, "get_8k_item", {"item": "5.02"}) == "CEO resigned."


async def test_raw_tools_need_a_10k(monkeypatch, llm):
    fake = FakeEdgar()
    fake.publish(eightk(E1, "2026-01-30"))
    fake.install(monkeypatch)

    out = await _ask(llm, "get_raw_risk_factors")
    assert out.startswith("Failed to retrieve risk factors") and "No 10-K filing found" in out


# ── #28: an analysis is saved under the filing its text was read from ────────


async def _ref_then_concurrent_refresh(edgar, monkeypatch) -> None:
    """Request A gets the K1 ref; request B then moves the shared retriever
    to K2 before A reads the section text."""
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "3600")
    assert (await latest_filing(TICKER, "10-K")).accession == K1
    edgar.publish(tenk(K2, "2026-11-01", K2_RISK))
    await asyncio.to_thread(get_retriever(TICKER).refresh_latest_filing, "10-K")


async def test_section_analysis_is_keyed_by_the_filing_read(fresh_db, edgar, llm, monkeypatch):
    await _ref_then_concurrent_refresh(edgar, monkeypatch)

    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert K2_RISK in llm.prompts[0]
    assert out["filing_reference"]["accession"] == K2
    assert out["filing_reference"]["filing_date"] == "2026-11-01"
    assert await _row("10-K", K2, "risk_10k") is not None
    assert await _row("10-K", K1, "risk_10k") is None


async def test_balance_sheet_analysis_is_keyed_by_the_filing_read(fresh_db, edgar, llm, monkeypatch):
    await _ref_then_concurrent_refresh(edgar, monkeypatch)

    out = json.loads(await _ask(llm, "get_balance_sheet_summary"))

    assert out["filing_reference"]["accession"] == K2
    assert await _row("10-K", K2, "balance") is not None
    assert await _row("10-K", K1, "balance") is None


async def test_8k_analysis_is_keyed_by_the_filing_read(fresh_db, edgar, llm, monkeypatch):
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "3600")
    assert (await latest_filing(TICKER, "8-K")).accession == E1
    edgar.publish(eightk(E2, "2026-11-10", earnings=False, item_text="CEO resigned."))
    await asyncio.to_thread(get_retriever(TICKER).refresh_latest_filing, "8-K")

    out = json.loads(await _ask(llm, "analyze_latest_8k"))

    assert "CEO resigned." in llm.prompts[0]
    assert out["filing_reference"]["accession"] == E2
    assert await _row("8-K", E2, "event") is not None
    assert await _row("8-K", E1, "earnings") is None
    assert await _row("8-K", E1, "event") is None


# ── #30: a stored analysis is served without downloading the filing ──────────


async def test_fresh_process_serves_stored_10k_and_8k_analyses_without_loading_them(fresh_db, edgar, llm):
    for name in ("get_risk_factors_summary", "get_balance_sheet_summary", "analyze_latest_8k"):
        await _ask(llm, name)
    assert llm.calls == 3
    tenk_filing, eightk_filing = edgar.filings["10-K"][0], edgar.filings["8-K"][0]
    loads = (tenk_filing.obj_calls, eightk_filing.obj_calls)

    _restart_process()
    for name in ("get_risk_factors_summary", "get_balance_sheet_summary", "analyze_latest_8k"):
        assert "filing_reference" in json.loads(await _ask(llm, name))

    assert llm.calls == 3
    assert (tenk_filing.obj_calls, eightk_filing.obj_calls) == loads


async def test_without_a_10q_the_stored_10k_analysis_is_served_unread(fresh_db, monkeypatch, llm):
    fake = FakeEdgar()
    fake.publish(tenk(K1, "2025-11-01", K1_RISK))
    fake.install(monkeypatch)
    stored = {"summary": "cached by the company page"}
    await db_module.save_filing_analysis(TICKER, "10-K", K1, "mda_10k", json.dumps(stored))

    out = json.loads(await _ask(llm, "get_mda_summary"))

    assert out["summary"] == "cached by the company page"
    assert fake.filings["10-K"][0].obj_calls == 0


async def test_boilerplate_10q_row_is_not_served_in_place_of_the_10k(fresh_db, edgar, llm):
    # The company page stores risk_10q even for boilerplate 10-Q text; chat
    # still prefers the 10-K's risk factors in that case.
    await db_module.save_filing_analysis(
        TICKER, "10-Q", Q1, "risk_10q", json.dumps({"summary": "boilerplate 10-Q"}),
    )

    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert out["filing_reference"]["accession"] == K1
    assert K1_RISK in llm.prompts[0]


# ── Registration ──────────────────────────────────────────────────────────────


def test_every_sec_tool_is_async_and_the_factory_returns_just_the_tools(llm):
    tools = create_sec_tools(TICKER, llm)
    assert isinstance(tools, list)
    for tool in tools:
        assert tool.func is None and tool.coroutine is not None, tool.name
