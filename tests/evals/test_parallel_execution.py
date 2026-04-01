"""Tests for parallel step execution and parallel get_all_summaries.

Marker: eval_unit — no API keys needed, all tools are mocked.

These tests verify two performance optimizations:

1. Parallel step executor (sec_graph.py):
   - Groups steps into dependency layers
   - Runs independent steps concurrently within each layer
   - Respects depends_on ordering between layers

2. Parallel get_all_summaries (sec_tools.py):
   - Runs risk, MD&A, and balance sheet summaries concurrently
   - Produces the same output structure as the sequential version
"""

import time

import pytest

from agents.graph.analyst_graph import (
    create_step_executor_node,
    _build_dependency_layers,
)
from agents.planner import QueryPlan, AnalysisStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "current_step_index": 0,
        "step_results": {},
        "final_response": "",
    }


def _slow_tool(delay: float = 0.2):
    """Return a tool function that sleeps for `delay` seconds."""
    def tool(query=""):
        time.sleep(delay)
        return f"result after {delay}s"
    return tool


# ===========================================================================
# _build_dependency_layers tests
# ===========================================================================


@pytest.mark.eval_unit
class TestDependencyLayers:
    """Test that _build_dependency_layers groups steps correctly."""

    def test_no_dependencies_single_layer(self):
        """Steps with no depends_on should all land in layer 0."""
        steps = [_make_step(1, "tool_a"), _make_step(2, "tool_b"), _make_step(3, "tool_c")]
        layers = _build_dependency_layers(steps)

        assert len(layers) == 1
        assert {s.id for s in layers[0]} == {1, 2, 3}

    def test_linear_chain(self):
        """A → B → C should produce three layers, one step each."""
        steps = [
            _make_step(1, "tool_a"),
            _make_step(2, "tool_b", depends_on=[1]),
            _make_step(3, "tool_c", depends_on=[2]),
        ]
        layers = _build_dependency_layers(steps)

        assert len(layers) == 3
        assert layers[0][0].id == 1
        assert layers[1][0].id == 2
        assert layers[2][0].id == 3

    def test_diamond_dependency(self):
        """Diamond: A → (B, C) → D. B and C should be in the same layer."""
        steps = [
            _make_step(1, "tool_a"),
            _make_step(2, "tool_b", depends_on=[1]),
            _make_step(3, "tool_c", depends_on=[1]),
            _make_step(4, "tool_d", depends_on=[2, 3]),
        ]
        layers = _build_dependency_layers(steps)

        assert len(layers) == 3
        assert {s.id for s in layers[0]} == {1}
        assert {s.id for s in layers[1]} == {2, 3}
        assert {s.id for s in layers[2]} == {4}

    def test_empty_steps(self):
        """No steps should return no layers."""
        assert _build_dependency_layers([]) == []


# ===========================================================================
# Parallel step executor tests
# ===========================================================================


@pytest.mark.eval_unit
class TestParallelStepExecutor:
    """Test that the step executor runs independent steps concurrently."""

    def test_all_steps_executed(self):
        """Every step should have a result after execution."""
        tools_dict = {
            "tool_a": lambda q="": "result_a",
            "tool_b": lambda q="": "result_b",
            "tool_c": lambda q="": "result_c",
        }
        plan = _make_plan([
            _make_step(1, "tool_a"),
            _make_step(2, "tool_b"),
            _make_step(3, "tool_c"),
        ])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "TEST")
        result = executor(state)

        assert result["step_results"] == {1: "result_a", 2: "result_b", 3: "result_c"}
        assert result["current_step_index"] == 3

    def test_independent_steps_run_concurrently(self):
        """Three independent slow tools should complete in ~1x delay, not ~3x.

        Each tool sleeps 0.2s. Sequential would take ~0.6s, parallel ~0.2s.
        We assert total time < 0.5s to leave margin for thread overhead.
        """
        tools_dict = {
            "tool_a": _slow_tool(0.2),
            "tool_b": _slow_tool(0.2),
            "tool_c": _slow_tool(0.2),
        }
        plan = _make_plan([
            _make_step(1, "tool_a"),
            _make_step(2, "tool_b"),
            _make_step(3, "tool_c"),
        ])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "TEST")
        start = time.monotonic()
        result = executor(state)
        elapsed = time.monotonic() - start

        assert len(result["step_results"]) == 3
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — steps likely ran sequentially"

    def test_dependency_ordering_respected(self):
        """Steps in later layers should only run after earlier layers complete.

        Layer 0: tool_a (records timestamp)
        Layer 1: tool_b depends on tool_a (records timestamp)

        tool_b's timestamp must be >= tool_a's.
        """
        timestamps: dict[str, float] = {}

        def make_timestamped_tool(name: str):
            def tool(query=""):
                time.sleep(0.1)
                timestamps[name] = time.monotonic()
                return f"{name} done"
            return tool

        tools_dict = {
            "tool_a": make_timestamped_tool("tool_a"),
            "tool_b": make_timestamped_tool("tool_b"),
        }
        plan = _make_plan([
            _make_step(1, "tool_a"),
            _make_step(2, "tool_b", depends_on=[1]),
        ])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "TEST")
        executor(state)

        assert timestamps["tool_b"] >= timestamps["tool_a"]

    def test_missing_tool_handled(self):
        """A missing tool should produce an error string, not crash the batch."""
        tools_dict = {"tool_a": lambda q="": "ok"}
        plan = _make_plan([
            _make_step(1, "tool_a"),
            _make_step(2, "missing_tool"),
        ])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "TEST")
        result = executor(state)

        assert result["step_results"][1] == "ok"
        assert "[ERROR" in result["step_results"][2]

    def test_exception_in_tool_handled(self):
        """A tool that raises should produce an error string, not crash."""
        def failing_tool(query=""):
            raise RuntimeError("boom")

        tools_dict = {"tool_a": failing_tool}
        plan = _make_plan([_make_step(1, "tool_a")])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "TEST")
        result = executor(state)

        assert "[ERROR: boom]" in result["step_results"][1]


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
