"""Tests for the Send-based fan-out worker and parallel get_all_summaries.

Marker: eval_unit — no API keys needed, all tools are mocked.

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
    """Wrap a callable as a langchain Tool with an ``.invoke`` method.

    The step executor calls ``tools_dict[name].invoke("")`` (rather than calling
    the func directly) so LangChain's tracing context propagates to any LLM
    calls inside. Tests must therefore expose ``Tool``-shaped objects, not raw
    lambdas — wrapping via ``Tool.from_function`` mirrors how production code
    builds the tools dict.
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
    """The worker runs a single step and returns {step_results: {id: StepResult}}."""

    def test_returns_step_result_for_prose_tool(self):
        tools_dict = {"tool_a": _as_tool("tool_a", lambda q="": "result_a")}
        worker = create_worker_node(tools_dict)

        delta = worker({"step": _make_step(1, "tool_a")})

        assert "step_results" in delta
        assert set(delta["step_results"].keys()) == {1}
        result = delta["step_results"][1]
        assert result["tool"] == "tool_a"
        assert result["raw"] == "result_a"
        assert result["data"] == {}
        assert result["filing_ref"] is None
        assert result["error"] is None

    def test_parses_json_tool_output(self):
        payload = (
            '{"filing_metadata": {"accession_number": "0000320193-25-000001",'
            ' "form_type": "8-K", "filing_date": "2025-04-01"},'
            ' "event_type": "earnings_release"}'
        )
        tools_dict = {"json_tool": _as_tool("json_tool", lambda q="": payload)}
        worker = create_worker_node(tools_dict)

        delta = worker({"step": _make_step(7, "json_tool")})
        result = delta["step_results"][7]

        assert result["data"]["event_type"] == "earnings_release"
        assert result["filing_ref"] == "0000320193-25-000001"
        assert result["raw"] == payload
        assert result["error"] is None

    def test_handles_non_json_tool_output(self):
        """Prose-returning tools land in raw with empty data."""
        tools_dict = {"tool_a": _as_tool("tool_a", lambda q="": "AAPL is at $185")}
        worker = create_worker_node(tools_dict)

        result = worker({"step": _make_step(1, "tool_a")})["step_results"][1]
        assert result["raw"] == "AAPL is at $185"
        assert result["data"] == {}
        assert result["error"] is None

    def test_handles_tool_exception(self):
        def boom(query=""):
            raise RuntimeError("boom")

        tools_dict = {"tool_a": _as_tool("tool_a", boom)}
        worker = create_worker_node(tools_dict)

        result = worker({"step": _make_step(1, "tool_a")})["step_results"][1]
        assert "[ERROR: boom]" in result["raw"]
        assert result["error"] == "boom"

    def test_missing_tool_returns_error_step_result(self):
        worker = create_worker_node({})
        result = worker({"step": _make_step(1, "ghost_tool")})["step_results"][1]
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

    def test_workers_run_in_parallel(self):
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
        final = compiled.invoke(_make_state(plan))
        elapsed = time.monotonic() - start

        assert set(final["step_results"].keys()) == {1, 2, 3}
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — workers likely ran sequentially"

    def test_reducer_unions_concurrent_writes(self):
        """All workers' step_results land in state via the merge reducer."""
        tools_dict = {
            "tool_a": _as_tool("tool_a", lambda q="": "ra"),
            "tool_b": _as_tool("tool_b", lambda q="": "rb"),
        }
        plan = _make_plan([_make_step(1, "tool_a"), _make_step(2, "tool_b")])
        compiled = self._compile_minimal_graph(tools_dict)

        final = compiled.invoke(_make_state(plan))

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


@pytest.mark.eval_unit
class TestParallelAllSummaries:
    """Test that _tool_all_summaries runs its three sub-calls concurrently."""

    def test_output_structure_preserved(self, mocker):
        """Output should contain all three section headers regardless of execution order."""
        mocker.patch("agents.tools.sec_tools._tool_risk_factors_summary", return_value="risk data")
        mocker.patch("agents.tools.sec_tools._tool_mda_summary", return_value="mda data")
        mocker.patch("agents.tools.sec_tools._tool_balance_sheet_summary", return_value="balance data")

        from agents.tools.sec_tools import _tool_all_summaries

        result = _tool_all_summaries("TEST", None, "")

        assert "=== RISK ANALYSIS ===" in result
        assert "=== MANAGEMENT OUTLOOK ===" in result
        assert "=== FINANCIAL HEALTH ===" in result
        assert "risk data" in result
        assert "mda data" in result
        assert "balance data" in result

    def test_runs_concurrently(self, mocker):
        """Three slow mocked summaries should complete in ~1x delay, not ~3x."""
        def slow_summary(ticker, llm, sec_header=""):
            time.sleep(0.2)
            return "summary"

        mocker.patch("agents.tools.sec_tools._tool_risk_factors_summary", side_effect=slow_summary)
        mocker.patch("agents.tools.sec_tools._tool_mda_summary", side_effect=slow_summary)
        mocker.patch("agents.tools.sec_tools._tool_balance_sheet_summary", side_effect=slow_summary)

        from agents.tools.sec_tools import _tool_all_summaries

        start = time.monotonic()
        result = _tool_all_summaries("TEST", None, "")
        elapsed = time.monotonic() - start

        assert "summary" in result
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — summaries likely ran sequentially"
