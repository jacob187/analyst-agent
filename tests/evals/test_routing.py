"""Pure-logic tests for the two routing functions in sec_graph.py.

Marker: eval_unit — NO API keys needed. These test the deterministic routing
logic by constructing state dicts directly and calling the routing functions.

`route_by_complexity` decides: simple → react_agent, anything else → planner.
`check_more_steps` decides: more steps left → step_executor, done → synthesizer.
"""

import pytest

from agents.graph.sec_graph import route_by_complexity, check_more_steps
from agents.planner import QueryPlan, AnalysisStep


# ---------------------------------------------------------------------------
# Helpers — build minimal state dicts for routing functions
# ---------------------------------------------------------------------------


def _make_state(complexity: str = "simple", **overrides):
    """Build a minimal AnalysisState dict for route_by_complexity."""
    state = {
        "messages": [],
        "ticker": "AAPL",
        "query_complexity": complexity,
        "classification": None,
        "plan": None,
        "current_step_index": 0,
        "step_results": {},
        "final_response": "",
    }
    state.update(overrides)
    return state


def _make_plan(num_steps: int) -> QueryPlan:
    """Create a QueryPlan with `num_steps` dummy steps."""
    steps = [
        AnalysisStep(
            id=i,
            action=f"Step {i}",
            tool="get_stock_info",
            rationale="test",
        )
        for i in range(1, num_steps + 1)
    ]
    return QueryPlan(
        query_type="complex",
        requires_planning=True,
        steps=steps,
        synthesis_approach="combine",
    )


# ---------------------------------------------------------------------------
# route_by_complexity tests
# ---------------------------------------------------------------------------


class TestRouteByComplexity:
    """route_by_complexity routes simple → react_agent, else → planner."""

    @pytest.mark.eval_unit
    def test_simple_routes_to_react(self):
        state = _make_state("simple")
        assert route_by_complexity(state) == "react_agent"

    @pytest.mark.eval_unit
    def test_complex_routes_to_planner(self):
        state = _make_state("complex")
        assert route_by_complexity(state) == "planner"

    @pytest.mark.eval_unit
    def test_moderate_routes_to_planner(self):
        """Moderate is NOT simple, so it should route to planner."""
        state = _make_state("moderate")
        assert route_by_complexity(state) == "planner"

    @pytest.mark.eval_unit
    def test_empty_string_routes_to_planner(self):
        """Empty string is not 'simple', so default route is planner."""
        state = _make_state("")
        assert route_by_complexity(state) == "planner"


# ---------------------------------------------------------------------------
# check_more_steps tests
# ---------------------------------------------------------------------------


class TestCheckMoreSteps:
    """check_more_steps loops step_executor until plan exhausted, then synthesizer."""

    @pytest.mark.eval_unit
    def test_more_steps_remaining(self):
        """When current_step_index < len(steps), keep executing."""
        plan = _make_plan(3)
        state = _make_state(plan=plan, current_step_index=0)
        assert check_more_steps(state) == "step_executor"

    @pytest.mark.eval_unit
    def test_midway_through_plan(self):
        """Halfway through a 3-step plan should still execute."""
        plan = _make_plan(3)
        state = _make_state(plan=plan, current_step_index=1)
        assert check_more_steps(state) == "step_executor"

    @pytest.mark.eval_unit
    def test_all_steps_done(self):
        """When all steps are done, route to synthesizer."""
        plan = _make_plan(3)
        state = _make_state(plan=plan, current_step_index=3)
        assert check_more_steps(state) == "synthesizer"

    @pytest.mark.eval_unit
    def test_empty_plan_steps(self):
        """Plan with no steps → synthesizer immediately."""
        plan = QueryPlan(
            query_type="simple",
            requires_planning=False,
            steps=[],
            synthesis_approach="direct",
        )
        state = _make_state(plan=plan, current_step_index=0)
        assert check_more_steps(state) == "synthesizer"

    @pytest.mark.eval_unit
    def test_none_plan(self):
        """No plan at all → synthesizer (graceful fallback)."""
        state = _make_state(plan=None, current_step_index=0)
        assert check_more_steps(state) == "synthesizer"
