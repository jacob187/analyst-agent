"""Tests for the Send-based fan-out worker and parallel get_all_summaries.

Marker: eval_unit — no API keys needed (fake tools; get_all_summaries runs the
real SEC tools over an in-memory EDGAR with a fake LLM).

These tests verify two performance/architecture properties:

1. Send-based fan-out (analyst_graph.py):
   - dispatch_steps emits one Send per plan step (depends_on intentionally ignored).
   - The worker node returns a typed StepResult merged via the step_results reducer.
   - End-to-end the compiled graph runs all workers in a single superstep.

2. Parallel get_all_summaries (sec_tools.py):
   - Runs risk, MD&A, and balance sheet summaries concurrently.
"""

import time

import pytest
from langchain_core.tools import Tool
from langgraph.types import Send

from agents.graph.analyst_graph import (
    create_worker_node,
    dispatch_steps,
    _extract_filing_ref,
    merge_step_results,
)
from agents.planner import QueryPlan, AnalysisStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_tool(name: str, func) -> Tool:
    """Wrap a callable as a langchain tool object.

    The worker awaits ``tools_dict[name].ainvoke("")`` (rather than calling the
    func directly) so the run's callbacks and tracing reach any LLM calls
    inside, so tests must expose tool objects, not raw lambdas. A single-input
    ``Tool`` is enough for the worker's empty-string call; production tools are
    ``StructuredTool``s, whose argument handling tests/test_tool_arg_schemas.py
    covers.
    """
    return Tool.from_function(name=name, description=name, func=func)


def _make_step(id: int, tool: str, depends_on: list[int] | None = None) -> AnalysisStep:
    return AnalysisStep(
        id=id,
        action=f"Execute {tool}",
        tool=tool,
        rationale="test",
        depends_on=depends_on or [],
    )


def _make_plan(steps: list[AnalysisStep]) -> QueryPlan:
    return QueryPlan(
        query_type="complex",
        requires_planning=True,
        steps=steps,
        synthesis_approach="combine",
    )


def _make_state(plan: QueryPlan) -> dict:
    return {
        "messages": [],
        "ticker": "TEST",
        "query_complexity": "complex",
        "classification": None,
        "plan": plan,
        "step_results": {},
        "final_response": "",
    }


def _slow_tool(name: str, delay: float = 0.2) -> Tool:
    """Return a Tool wrapping a function that sleeps for `delay` seconds."""
    def fn(query=""):
        time.sleep(delay)
        return f"result after {delay}s"
    return _as_tool(name, fn)


# ===========================================================================
# dispatch_steps tests
# ===========================================================================


@pytest.mark.eval_unit
class TestDispatchSteps:
    """dispatch_steps emits one Send('worker', {'step': s}) per plan step."""

    def test_emits_one_send_per_step(self):
        plan = _make_plan(
            [_make_step(1, "tool_a"), _make_step(2, "tool_b"), _make_step(3, "tool_c")]
        )
        sends = dispatch_steps({"plan": plan})

        assert len(sends) == 3
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "worker" for s in sends)
        assert [s.arg["step"].id for s in sends] == [1, 2, 3]

    def test_ignores_depends_on(self):
        """Phase-3 design choice: depends_on is preserved in AnalysisStep but
        not respected by dispatch — every step fans out into the same superstep."""
        plan = _make_plan(
            [
                _make_step(1, "tool_a"),
                _make_step(2, "tool_b", depends_on=[1]),
                _make_step(3, "tool_c", depends_on=[2]),
            ]
        )
        sends = dispatch_steps({"plan": plan})
        assert len(sends) == 3
        assert all(s.node == "worker" for s in sends)

    def test_empty_plan_routes_to_synthesizer(self):
        empty_plan = QueryPlan(
            query_type="simple",
            requires_planning=False,
            steps=[],
            synthesis_approach="direct",
        )
        assert dispatch_steps({"plan": empty_plan}) == ["synthesizer"]

    def test_no_plan_routes_to_synthesizer(self):
        assert dispatch_steps({"plan": None}) == ["synthesizer"]


# ===========================================================================
# worker node tests (Send payload → StepResult delta)
# ===========================================================================


@pytest.mark.eval_unit
class TestWorkerNode:
    """The worker runs a single step and returns {step_results: {id: StepResult}}.

    The worker is `async def` and uses `tool.ainvoke(...)`. LangChain auto-bridges
    sync tools (`func=`) to a thread pool, so the same Tool wrappers work
    unchanged — only the dispatch surface is async now.
    """

    async def test_returns_step_result_for_prose_tool(self):
        tools_dict = {"tool_a": _as_tool("tool_a", lambda q="": "result_a")}
        worker = create_worker_node(tools_dict)

        delta = await worker({"step": _make_step(1, "tool_a")})

        assert "step_results" in delta
        assert set(delta["step_results"].keys()) == {1}
        result = delta["step_results"][1]
        assert result["tool"] == "tool_a"
        assert result["raw"] == "result_a"
        assert result["data"] == {}
        assert result["filing_ref"] is None
        assert result["error"] is None

    async def test_parses_json_tool_output(self):
        payload = (
            '{"filing_metadata": {"accession_number": "0000320193-25-000001",'
            ' "form_type": "8-K", "filing_date": "2025-04-01"},'
            ' "event_type": "earnings_release"}'
        )
        tools_dict = {"json_tool": _as_tool("json_tool", lambda q="": payload)}
        worker = create_worker_node(tools_dict)

        delta = await worker({"step": _make_step(7, "json_tool")})
        result = delta["step_results"][7]

        assert result["data"]["event_type"] == "earnings_release"
        assert result["filing_ref"] == "0000320193-25-000001"
        assert result["raw"] == payload
        assert result["error"] is None

    async def test_handles_non_json_tool_output(self):
        """Prose-returning tools land in raw with empty data."""
        tools_dict = {"tool_a": _as_tool("tool_a", lambda q="": "AAPL is at $185")}
        worker = create_worker_node(tools_dict)

        result = (await worker({"step": _make_step(1, "tool_a")}))["step_results"][1]
        assert result["raw"] == "AAPL is at $185"
        assert result["data"] == {}
        assert result["error"] is None

    async def test_handles_tool_exception(self):
        def boom(query=""):
            raise RuntimeError("boom")

        tools_dict = {"tool_a": _as_tool("tool_a", boom)}
        worker = create_worker_node(tools_dict)

        result = (await worker({"step": _make_step(1, "tool_a")}))["step_results"][1]
        assert "[ERROR: boom]" in result["raw"]
        assert result["error"] == "boom"

    async def test_missing_tool_returns_error_step_result(self):
        worker = create_worker_node({})
        result = (await worker({"step": _make_step(1, "ghost_tool")}))["step_results"][1]
        assert "[ERROR" in result["raw"]
        assert result["error"] == "Tool 'ghost_tool' not found"


# ===========================================================================
# End-to-end fan-out: compiled graph runs N workers in one superstep
# ===========================================================================


@pytest.mark.eval_unit
class TestSendFanOutEndToEnd:
    """Compile a minimal StateGraph and verify Send fan-out actually parallelizes."""

    def _compile_minimal_graph(self, tools_dict):
        """Build a tiny StateGraph: planner_stub → dispatch → worker → END.

        Mirrors the relevant slice of create_planning_workflow without the
        router/react/synthesizer branches we don't need for this test.
        """
        from langgraph.graph import StateGraph, END
        from agents.graph.analyst_graph import AnalysisState

        graph = StateGraph(AnalysisState)
        graph.add_node("planner_stub", lambda state: state)
        graph.add_node("worker", create_worker_node(tools_dict))
        graph.set_entry_point("planner_stub")
        graph.add_conditional_edges("planner_stub", dispatch_steps, ["worker", END])
        graph.add_edge("worker", END)
        return graph.compile()

    async def test_workers_run_in_parallel(self):
        """Three Send-fanned workers should complete in ~1x delay, not ~3x."""
        tools_dict = {
            "tool_a": _slow_tool("tool_a", 0.2),
            "tool_b": _slow_tool("tool_b", 0.2),
            "tool_c": _slow_tool("tool_c", 0.2),
        }
        plan = _make_plan(
            [_make_step(1, "tool_a"), _make_step(2, "tool_b"), _make_step(3, "tool_c")]
        )
        compiled = self._compile_minimal_graph(tools_dict)

        start = time.monotonic()
        final = await compiled.ainvoke(_make_state(plan))
        elapsed = time.monotonic() - start

        assert set(final["step_results"].keys()) == {1, 2, 3}
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — workers likely ran sequentially"

    async def test_reducer_unions_concurrent_writes(self):
        """All workers' step_results land in state via the merge reducer."""
        tools_dict = {
            "tool_a": _as_tool("tool_a", lambda q="": "ra"),
            "tool_b": _as_tool("tool_b", lambda q="": "rb"),
        }
        plan = _make_plan([_make_step(1, "tool_a"), _make_step(2, "tool_b")])
        compiled = self._compile_minimal_graph(tools_dict)

        final = await compiled.ainvoke(_make_state(plan))

        assert final["step_results"][1]["raw"] == "ra"
        assert final["step_results"][2]["raw"] == "rb"


# ===========================================================================
# StepResult plumbing: reducer and filing_ref extractor
# ===========================================================================


@pytest.mark.eval_unit
class TestMergeStepResults:
    """The reducer merges parallel step writes by unioning unique step_id keys."""

    def test_unions_disjoint_keys(self):
        left = {1: {"tool": "a", "data": {}, "raw": "ra", "filing_ref": None, "error": None}}
        right = {2: {"tool": "b", "data": {}, "raw": "rb", "filing_ref": None, "error": None}}

        merged = merge_step_results(left, right)

        assert set(merged.keys()) == {1, 2}
        assert merged[1]["raw"] == "ra"
        assert merged[2]["raw"] == "rb"

    def test_empty_inputs(self):
        assert merge_step_results({}, {}) == {}

    def test_right_wins_on_collision(self):
        """If a key appears on both sides (shouldn't in practice), right wins.

        We don't rely on this because each worker writes a unique step_id,
        but the dict-union semantics need to be predictable.
        """
        left = {1: {"tool": "a", "data": {}, "raw": "old", "filing_ref": None, "error": None}}
        right = {1: {"tool": "a", "data": {}, "raw": "new", "filing_ref": None, "error": None}}

        assert merge_step_results(left, right)[1]["raw"] == "new"


@pytest.mark.eval_unit
class TestExtractFilingRef:
    """Filing-ref priority: accession_number > form_type:filing_date > filing_date > None."""

    def test_accession_wins(self):
        data = {
            "filing_metadata": {
                "accession_number": "0000320193-25-000001",
                "form_type": "8-K",
                "filing_date": "2025-04-01",
            }
        }
        assert _extract_filing_ref(data) == "0000320193-25-000001"

    def test_form_and_date_fallback(self):
        data = {"filing_metadata": {"form_type": "8-K", "filing_date": "2025-04-01"}}
        assert _extract_filing_ref(data) == "8-K:2025-04-01"

    def test_date_only_fallback(self):
        data = {"filing_metadata": {"filing_date": "2025-04-01"}}
        assert _extract_filing_ref(data) == "2025-04-01"

    def test_no_metadata_returns_none(self):
        assert _extract_filing_ref({"event_type": "earnings"}) is None

    def test_empty_data_returns_none(self):
        assert _extract_filing_ref({}) is None

    def test_non_dict_metadata_returns_none(self):
        assert _extract_filing_ref({"filing_metadata": "not a dict"}) is None


# ===========================================================================
# Parallel get_all_summaries tests
# ===========================================================================


@pytest.fixture
async def sec_env(tmp_path, monkeypatch):
    """Temp DB (the analysis cache's store) + an in-memory EDGAR; yields a fake
    chat model at the LLM boundary that records its calls."""
    import api.db as db_module
    from tests.sec_fakes import CountingChatModel, FakeEdgar, eightk, tenk, tenq

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    monkeypatch.delenv("FILING_FRESHNESS_TTL_SECONDS", raising=False)
    await db_module.init_db()
    edgar = FakeEdgar()
    edgar.publish(tenk("0000320193-25-000101", "2025-11-01", "10-K risk: supply chain."))
    edgar.publish(tenq("0000320193-26-000111", "2026-02-01", "10-Q risk: new tariffs."))
    edgar.publish(eightk("0000320193-26-000121", "2026-01-30"))
    edgar.install(monkeypatch)
    yield CountingChatModel()
    await db_module.close_db()


async def _all_summaries(llm) -> str:
    from agents.tools.sec_tools import create_sec_tools

    tools = {t.name: t for t in create_sec_tools("TEST", llm)}
    return await tools["get_all_summaries"].ainvoke("")


@pytest.mark.eval_unit
class TestParallelAllSummaries:
    """get_all_summaries runs its three analyses concurrently (real tools over
    a fake EDGAR; only the LLM is faked)."""

    async def test_output_structure_preserved(self, sec_env):
        """Each section header is followed by its own analysis."""
        result = await _all_summaries(sec_env)

        risk, mda, balance = (
            result.split(header, 1)[1]
            for header in ("=== RISK ANALYSIS ===", "=== MANAGEMENT OUTLOOK ===", "=== FINANCIAL HEALTH ===")
        )
        assert "risk analysis" in risk.split("===", 1)[0]
        assert "mda analysis" in mda.split("===", 1)[0]
        assert "balance analysis" in balance
        assert sec_env.calls == 3

    async def test_runs_concurrently(self, sec_env):
        """All three LLM calls are in flight at once, not one after another."""
        sec_env.delay = 0.2

        await _all_summaries(sec_env)

        assert sec_env.peak_in_flight == 3
