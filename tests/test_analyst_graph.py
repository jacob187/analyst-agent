"""Regression tests for graph-level invariants in `agents/graph/analyst_graph.py`.

The plan called for truncating `state["messages"]` before the synthesizer to
bound per-turn token cost. Verification of the current code revealed the
synthesizer prompt is built from `plan` + `step_results` + the *latest* query
only (via `_get_latest_query`), so prompt size is already independent of
history length. No truncation code is needed — but we lock that invariant
here so a future refactor that wires `state["messages"]` into the prompt
fails this test instead of silently re-introducing the cost vector.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.graph.analyst_graph import _build_synthesis_prompt
from agents.planner import AnalysisStep, QueryPlan


def _state(messages: list) -> dict:
    plan = QueryPlan(
        query_type="complex",
        requires_planning=True,
        steps=[
            AnalysisStep(
                id=1,
                action="Fetch risk factors",
                tool="get_risk_factors_summary",
                rationale="test",
            )
        ],
        synthesis_approach="combine",
    )
    return {
        "messages": messages,
        "ticker": "AAPL",
        "query_complexity": "complex",
        "classification": None,
        "plan": plan,
        "step_results": {
            1: {
                "tool": "get_risk_factors_summary",
                "data": {},
                "raw": "Some risk factor analysis text.",
                "filing_ref": None,
                "error": None,
            }
        },
        "conflicts": [],
        "final_response": "",
    }


@pytest.mark.eval_unit
class TestSynthesisPromptInvariant:
    def test_synthesis_prompt_size_independent_of_message_history(self):
        """Building the synthesis prompt over a long noisy history must produce
        the same string as building it over a one-message history with the same
        latest query — because the prompt only reads the latest query plus
        bounded plan/step_results.
        """
        query = "What are AAPL's key risks?"

        short_state = _state([HumanMessage(content=query)])
        # Last HumanMessage is the latest user query — keep it last so
        # `_get_latest_query` returns the same string in both states.
        noisy_state = _state(
            [HumanMessage(content="prior turn")]
            + [AIMessage(content="noise " * 1000) for _ in range(100)]
            + [HumanMessage(content=query)]
        )

        short_prompt = _build_synthesis_prompt(short_state, "AAPL")
        noisy_prompt = _build_synthesis_prompt(noisy_state, "AAPL")

        assert short_prompt == noisy_prompt, (
            "Synthesizer prompt changed when message history grew. "
            "The synthesizer must read only the latest query — adding "
            "state['messages'] to the prompt re-introduces the O(N) per-turn "
            "cost vector."
        )

    def test_synthesis_prompt_uses_latest_human_query(self):
        """Sanity: the prompt does include the latest user query verbatim."""
        state = _state(
            [
                HumanMessage(content="earlier question"),
                AIMessage(content="earlier answer"),
                HumanMessage(content="LATEST_USER_QUERY"),
            ]
        )
        prompt = _build_synthesis_prompt(state, "AAPL")
        assert "LATEST_USER_QUERY" in prompt
