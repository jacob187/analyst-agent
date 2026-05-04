"""Pure-logic tests for the routing function in analyst_graph.py.

Marker: eval_unit — NO API keys needed. These test the deterministic routing
logic by constructing state dicts directly and calling the routing function.

`route_by_complexity` decides: simple → react_agent, anything else → planner.
"""

import pytest

from agents.graph.analyst_graph import route_by_complexity, UNCLEAR_QUERY_RESPONSE


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
        "step_results": {},
        "final_response": "",
    }
    state.update(overrides)
    return state


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
    def test_unclear_routes_to_end(self):
        """Unclear queries should route to __end__ — no tools invoked."""
        state = _make_state("unclear", final_response=UNCLEAR_QUERY_RESPONSE)
        assert route_by_complexity(state) == "__end__"

    @pytest.mark.eval_unit
    def test_unclear_state_has_response(self):
        """When routed as unclear, final_response should contain a clarification."""
        state = _make_state("unclear", final_response=UNCLEAR_QUERY_RESPONSE)
        assert "rephrase" in state["final_response"].lower()
        assert route_by_complexity(state) == "__end__"

    @pytest.mark.eval_unit
    def test_empty_string_routes_to_planner(self):
        """Empty string is not 'simple', so default route is planner."""
        state = _make_state("")
        assert route_by_complexity(state) == "planner"
