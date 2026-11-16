"""Shared SEC analysis cache (agents/sec_workflow/analysis_cache.py) as used by
the chat SEC tools and the company filings endpoint.

Real temp SQLite, the real SECDataRetrieval over an in-memory EDGAR
(tests/sec_fakes.py), and a chat model that counts calls at the LLM boundary.
"""

import asyncio
import json
from datetime import datetime

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api.db as db_module
from agents.model_registry import get_default_model
from agents.sec_workflow import analysis_cache
from agents.sec_workflow.analysis_cache import (
    DEFAULT_FRESHNESS_TTL_SECONDS,
    FilingRef,
    freshness_ttl_seconds,
    get_or_analyze,
    latest_filing,
)
from agents.tools.sec_tools import create_sec_tools
from api.llm_concurrency import LLMCallCharger
from tests.conftest import byok_header
from tests.sec_fakes import CountingChatModel, FakeEdgar, eightk, tenk, tenq

TICKER = "TSTC"
K1, K2 = "0000320193-25-000101", "0000320193-26-000202"
Q1 = "0000320193-26-000111"
E1, E2 = "0000320193-26-000121", "0000320193-26-000222"
TENK_RISK_1 = "TENK-RISK-1: supply chain concentration and antitrust exposure."
TENK_RISK_2 = "TENK-RISK-2: new export controls on advanced chips."
# Boilerplate, so the risk tools fall back to the 10-K (key risk_10k).
TENQ_RISK = "There have been no material changes to the risk factors previously disclosed."
USER = "user_analysiscache"


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
    fake.publish(tenk(K1, "2025-11-01", TENK_RISK_1))
    fake.publish(tenq(Q1, "2026-02-01", TENQ_RISK, mda="TENQ-MDA-1: services revenue grew 12%."))
    fake.publish(eightk(E1, "2026-01-30", earnings=True))
    return fake.install(monkeypatch)


@pytest.fixture
def llm():
    return CountingChatModel()


def _tools(llm):
    tools = create_sec_tools(TICKER, llm)
    return {t.name: t for t in tools}


async def _ask(llm, name: str, config=None) -> str:
    return await _tools(llm)[name].ainvoke("", config=config)


def _stored_part(served: dict) -> dict:
    return {k: v for k, v in served.items() if k != "filing_reference"}


async def _row(form: str, accession: str, analysis_type: str) -> dict | None:
    row = await db_module.get_filing_analysis(TICKER, form, accession, analysis_type)
    return json.loads(row["analysis_json"]) if row else None


def _restart_process() -> None:
    """Drop every in-process cache, as a restart of the sleeping host would."""
    for cache in (
        analysis_cache._shared_retrievers,
        analysis_cache._latest_filings,
        analysis_cache._latest_locks,
        analysis_cache._analysis_locks,
    ):
        cache.clear()


# ── AC1 / AC2: hit serves without LLM; miss generates once under the page's key ──


async def test_cached_analysis_served_without_llm_call(fresh_db, edgar, llm):
    stored = {"summary": "cached by the company page", "key_risks": ["x"]}
    await db_module.save_filing_analysis(TICKER, "10-K", K1, "risk_10k", json.dumps(stored))

    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert llm.calls == 0
    assert _stored_part(out) == stored


async def test_cache_miss_generates_once_and_saves_under_company_page_key(fresh_db, edgar, llm):
    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert llm.calls == 1
    assert TENK_RISK_1 in llm.prompts[0]
    assert await _row("10-K", K1, "risk_10k") == _stored_part(out)

    await _ask(llm, "get_risk_factors_summary")
    assert llm.calls == 1


async def test_substantive_10q_section_is_keyed_to_the_10q(fresh_db, edgar, llm):
    out = json.loads(await _ask(llm, "get_mda_summary"))

    assert "TENQ-MDA-1" in llm.prompts[0]
    assert out["filing_reference"]["accession"] == Q1
    assert await _row("10-Q", Q1, "mda_10q") == _stored_part(out)


async def test_concurrent_identical_misses_generate_once(fresh_db, edgar, llm):
    llm.delay = 0.05
    first, second = await asyncio.gather(
        _ask(llm, "get_risk_factors_summary"), _ask(llm, "get_risk_factors_summary"),
    )
    assert llm.calls == 1
    assert _stored_part(json.loads(first)) == _stored_part(json.loads(second))


async def test_failed_analysis_is_not_saved(fresh_db, edgar):
    broken = FakeListChatModel(responses=["this is not JSON"])
    out = await _tools(broken)["get_risk_factors_summary"].ainvoke("")

    assert out.startswith("Failed to analyze risk factors")
    assert await _row("10-K", K1, "risk_10k") is None


async def test_works_uncached_when_db_is_unusable(tmp_path, monkeypatch, edgar, llm):
    # Tables never created (e.g. a script that skipped init_db): read and save
    # fail, the tool still answers.
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "uninitialised.db")
    monkeypatch.setattr(db_module, "_db", None)
    try:
        out = json.loads(await _ask(llm, "get_risk_factors_summary"))
    finally:
        await db_module.close_db()
    assert llm.calls == 1
    assert out["summary"] == "risk analysis #1"
    assert out["filing_reference"]["accession"] == K1


# ── AC4: a fresh process reuses the DB ─────────────────────────────────────────


async def test_second_question_in_fresh_process_makes_no_llm_calls(fresh_db, edgar, llm):
    await _ask(llm, "get_all_summaries")
    await _ask(llm, "analyze_latest_8k")
    assert llm.calls == 4

    _restart_process()
    after_restart = CountingChatModel()
    await _ask(after_restart, "get_all_summaries")
    await _ask(after_restart, "analyze_latest_8k")

    assert after_restart.calls == 0


# ── AC5: freshness check before serving ────────────────────────────────────────


async def test_no_newer_filing_serves_cached_analysis(fresh_db, edgar, llm, monkeypatch):
    await _ask(llm, "get_risk_factors_summary")
    requests = edgar.requests

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")  # TTL lapsed
    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert edgar.requests > requests  # EDGAR was asked
    assert llm.calls == 1
    assert out["filing_reference"]["accession"] == K1


async def test_newer_10k_is_analysed_and_cited(fresh_db, edgar, llm, monkeypatch):
    await _ask(llm, "get_risk_factors_summary")
    edgar.publish(tenk(K2, "2026-11-01", TENK_RISK_2))

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert llm.calls == 2
    assert TENK_RISK_2 in llm.prompts[1]  # the new filing's text, not the old one's
    assert out["filing_reference"]["accession"] == K2
    assert out["filing_reference"]["filing_date"] == "2026-11-01"
    assert await _row("10-K", K2, "risk_10k") == _stored_part(out)
    assert await _row("10-K", K1, "risk_10k") is not None


async def test_newer_8k_is_analysed_and_cited(fresh_db, edgar, llm, monkeypatch):
    first = json.loads(await _ask(llm, "analyze_latest_8k"))
    assert first["filing_reference"]["accession"] == E1
    edgar.publish(eightk(E2, "2026-11-10", earnings=False, item_text="CEO resigned effective today."))

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    out = json.loads(await _ask(llm, "analyze_latest_8k"))

    assert llm.calls == 2
    assert "CEO resigned effective today." in llm.prompts[1]
    assert out["event_type"] == "leadership_change"
    assert out["filing_reference"] == {
        "filing_date": "2026-11-10",
        "accession": E2,
        "analyzed_at": out["filing_reference"]["analyzed_at"],
    }
    assert await _row("8-K", E2, "event") == _stored_part(out)


async def test_edgar_outage_after_ttl_serves_the_stored_analyses(fresh_db, edgar, llm, monkeypatch):
    # The TTL lapsed and EDGAR is down: the stale ref still names the
    # retriever's filing, so its stored analysis is served (no LLM call).
    risk = json.loads(await _ask(llm, "get_risk_factors_summary"))
    eight = json.loads(await _ask(llm, "analyze_latest_8k"))
    edgar.outage = RuntimeError("EDGAR 429")
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")

    assert json.loads(await _ask(llm, "get_risk_factors_summary")) == risk
    assert json.loads(await _ask(llm, "analyze_latest_8k")) == eight
    assert llm.calls == 2


async def test_edgar_outage_with_nothing_cached_still_fails(fresh_db, edgar, llm):
    edgar.outage = RuntimeError("EDGAR 429")
    out = await _ask(llm, "get_risk_factors_summary")
    assert out.startswith("Failed to analyze risk factors") and "EDGAR 429" in out


async def test_event_8k_without_item_text_is_not_analysed(fresh_db, monkeypatch, llm):
    fake = FakeEdgar()
    fake.publish(eightk(E1, "2026-01-30", earnings=False, item_text=""))
    fake.install(monkeypatch)

    out = json.loads(await _ask(llm, "analyze_latest_8k"))

    assert llm.calls == 0
    assert out["found"] is False
    assert await _row("8-K", E1, "event") is None


async def test_event_8k_without_item_text_still_serves_cached_analysis(fresh_db, monkeypatch, llm):
    fake = FakeEdgar()
    fake.publish(eightk(E1, "2026-01-30", earnings=False, item_text=""))
    fake.install(monkeypatch)
    stored = {"summary": "cached event", "event_type": "leadership_change"}
    await db_module.save_filing_analysis(TICKER, "8-K", E1, "event", json.dumps(stored))

    out = json.loads(await _ask(llm, "analyze_latest_8k"))

    assert llm.calls == 0
    assert _stored_part(out) == stored


# ── AC6: every served analysis cites its filing ────────────────────────────────


async def test_served_analysis_carries_filing_reference(fresh_db, edgar, llm):
    generated = json.loads(await _ask(llm, "get_balance_sheet_summary"))
    cached = json.loads(await _ask(llm, "get_balance_sheet_summary"))

    ref = generated["filing_reference"]
    assert ref["filing_date"] == "2025-11-01"
    assert ref["accession"] == K1
    assert datetime.fromisoformat(ref["analyzed_at"]).tzinfo is not None
    assert cached["filing_reference"] == ref


async def test_filing_reference_is_not_stored(fresh_db, edgar, llm):
    await _ask(llm, "get_balance_sheet_summary")
    assert "filing_reference" not in await _row("10-K", K1, "balance")


async def test_get_or_analyze_miss_without_generator_returns_none(fresh_db):
    assert await get_or_analyze(TICKER, FilingRef("10-K", K1, "2025-11-01"), "risk_10k") is None


# ── AC7: latest-filing lookup cached per (ticker, form) for the TTL ────────────


async def test_latest_filing_lookup_cached_within_ttl(edgar, monkeypatch):
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "3600")
    first = await latest_filing(TICKER, "10-K")
    requests = edgar.requests
    edgar.publish(tenk(K2, "2026-11-01", TENK_RISK_2))

    assert await latest_filing(TICKER, "10-K") == first
    assert edgar.requests == requests

    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    assert (await latest_filing(TICKER, "10-K")).accession == K2
    assert edgar.requests > requests


async def test_one_lookup_serves_every_form(edgar):
    await latest_filing(TICKER, "10-K")
    built = edgar.companies
    assert (await latest_filing(TICKER, "10-Q")).accession == Q1
    assert (await latest_filing(TICKER, "8-K")).accession == E1
    assert edgar.companies == built


async def test_concurrent_first_lookups_share_one_sec_request(edgar):
    edgar.latency = 0.05
    refs = await asyncio.gather(*(latest_filing(TICKER, "10-K") for _ in range(5)))
    assert {r.accession for r in refs} == {K1}
    assert edgar.companies == 1


async def test_evicted_retriever_does_not_pair_old_ref_with_new_text(fresh_db, edgar, llm, monkeypatch):
    # The retriever LRU drops the ticker while its 10-K ref is still within
    # the TTL; a newer 10-K lands meanwhile. The replacement retriever loads
    # the new 10-K, so the ref must be looked up again — never key the new
    # filing's analysis under the old accession.
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "3600")
    await latest_filing(TICKER, "10-K")
    analysis_cache._shared_retrievers.clear()
    edgar.publish(tenk(K2, "2026-11-01", TENK_RISK_2))

    out = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert TENK_RISK_2 in llm.prompts[0]
    assert out["filing_reference"]["accession"] == K2
    assert await _row("10-K", K1, "risk_10k") is None


async def test_company_without_the_form_caches_none(monkeypatch):
    fake = FakeEdgar().install(monkeypatch)
    assert await latest_filing(TICKER, "10-K") is None
    assert await latest_filing(TICKER, "10-K") is None
    assert fake.companies == 1


def test_ttl_defaults_to_four_hours(monkeypatch):
    monkeypatch.delenv("FILING_FRESHNESS_TTL_SECONDS", raising=False)
    assert freshness_ttl_seconds() == DEFAULT_FRESHNESS_TTL_SECONDS == 4 * 3600


@pytest.mark.parametrize("raw", ["abc", "-5", ""])
def test_invalid_ttl_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", raw)
    assert freshness_ttl_seconds() == DEFAULT_FRESHNESS_TTL_SECONDS


async def test_unsupported_form_is_refused(edgar):
    with pytest.raises(ValueError, match="Unsupported form"):
        await latest_filing(TICKER, "S-1")
    assert edgar.requests == 0


# ── Chat charging: LLM calls run with the chat run's callbacks ─────────────────


async def test_all_summaries_llm_calls_reach_run_callbacks(fresh_db, edgar, llm):
    charger = LLMCallCharger(USER)
    out = await _ask(llm, "get_all_summaries", config={"callbacks": [charger]})

    assert "=== RISK ANALYSIS ===" in out and "balance analysis" in out
    assert llm.calls == 3
    assert await charger.settle() == 3
    assert await db_module.get_llm_usage(USER) == 3


async def test_cache_hits_charge_nothing(fresh_db, edgar, llm):
    await _ask(llm, "get_all_summaries")
    charger = LLMCallCharger(USER)
    await _ask(llm, "get_all_summaries", config={"callbacks": [charger]})
    assert await charger.settle() == 0


async def test_graph_worker_charges_the_runs_callbacks(fresh_db, edgar, llm):
    # The planner path: the step worker awaits `tool.ainvoke("")` with no
    # config of its own; the run's callbacks must still reach the SEC LLM calls.
    from langgraph.graph import END, StateGraph

    from agents.graph.analyst_graph import (
        AnalysisState, _build_tools_dict, create_worker_node, dispatch_steps,
    )
    from agents.planner import AnalysisStep, QueryPlan

    tools = create_sec_tools(TICKER, llm)
    graph = StateGraph(AnalysisState)
    graph.add_node("planner_stub", lambda state: state)
    graph.add_node("worker", create_worker_node(_build_tools_dict(tools)))
    graph.set_entry_point("planner_stub")
    graph.add_conditional_edges("planner_stub", dispatch_steps, ["worker", END])
    graph.add_edge("worker", END)
    step = AnalysisStep(id=1, action="all", tool="get_all_summaries", rationale="t", depends_on=[])
    plan = QueryPlan(query_type="complex", requires_planning=True, steps=[step], synthesis_approach="x")
    state = {
        "messages": [], "ticker": TICKER, "query_complexity": "complex",
        "classification": None, "plan": plan, "step_results": {}, "final_response": "",
    }

    charger = LLMCallCharger(USER)
    final = await graph.compile().ainvoke(state, config={"callbacks": [charger]})

    result = final["step_results"][1]
    raw = result["raw"] if isinstance(result, dict) else result.raw
    assert "balance analysis" in raw
    assert llm.calls == 3
    assert await charger.settle() == 3


async def test_llm_tools_are_async_only(llm):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tools = _tools(llm)
    for name in ("get_risk_factors_summary", "get_mda_summary", "get_balance_sheet_summary",
                 "get_all_summaries", "analyze_latest_8k"):
        assert tools[name].func is None and tools[name].coroutine is not None, name
        # Still bindable by the ReAct agent.
        assert convert_to_openai_tool(tools[name])["function"]["name"] == name


# ── AC3 / AC8: chat and company page share entries and the freshness check ────


@pytest.fixture
def page(monkeypatch, llm):
    """The company filings endpoint, generating with the same counting LLM."""
    from api.main import app

    monkeypatch.setattr("agents.llm_factory.create_llm", lambda model_id, api_key: llm)
    transport = httpx.ASGITransport(app=app)
    headers = {**byok_header(get_default_model().provider, "byok-test-key"), "X-User-Id": USER}

    async def get() -> dict:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/company/{TICKER}/filings", headers=headers)
        assert resp.status_code == 200, resp.text
        return resp.json()

    return get


async def test_chat_serves_the_company_pages_analysis(fresh_db, edgar, llm, page):
    data = await page()
    page_risk = data["tenk"]["risk_10k"]
    calls = llm.calls

    chat_risk = json.loads(await _ask(llm, "get_risk_factors_summary"))

    assert llm.calls == calls
    assert chat_risk == page_risk  # same stored analysis, same filing reference
    assert _stored_part(chat_risk) == await _row("10-K", K1, "risk_10k")


async def test_company_page_serves_chats_analysis(fresh_db, edgar, llm, page):
    chat_risk = json.loads(await _ask(llm, "get_risk_factors_summary"))
    chat_8k = json.loads(await _ask(llm, "analyze_latest_8k"))
    assert llm.calls == 2

    data = await page()

    risk_prompts = [p for p in llm.prompts[2:] if TENK_RISK_1 in p]
    assert risk_prompts == []  # the page did not re-analyse the 10-K risk factors
    assert data["tenk"]["risk_10k"] == chat_risk
    assert data["eightk"]["analysis"] == chat_8k


async def test_company_page_and_chat_share_the_freshness_lookup(fresh_db, edgar, llm, page, monkeypatch):
    await page()
    requests = edgar.requests

    # Within the TTL chat reuses the page's lookup: no EDGAR request.
    await _ask(llm, "get_risk_factors_summary")
    assert edgar.requests == requests

    # A newer 10-K after the TTL lapses: chat finds and analyses it ...
    edgar.publish(tenk(K2, "2026-11-01", TENK_RISK_2))
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "0")
    chat_risk = json.loads(await _ask(llm, "get_risk_factors_summary"))
    assert chat_risk["filing_reference"]["accession"] == K2

    # ... and the page, through the same lookup, reports and reuses it.
    monkeypatch.setenv("FILING_FRESHNESS_TTL_SECONDS", "3600")
    calls = llm.calls
    data = await page()
    assert edgar.requests > requests
    assert data["tenk"]["metadata"]["accession"] == K2
    assert data["tenk"]["risk_10k"] == chat_risk
    assert not any(TENK_RISK_2 in p for p in llm.prompts[calls:])
