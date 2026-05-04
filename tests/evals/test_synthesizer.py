"""Tests for the synthesizer node (create_synthesizer_node).

Marker: eval_fast — each test makes a single LLM call to synthesize results.

The synthesizer takes step_results from all executed steps and produces a
final_response that should reference content from ALL results, not just the
last one. This is a common LLM failure mode: recency bias causes the model
to focus on the most recent input and ignore earlier steps.
"""

import pytest

from langchain_core.messages import HumanMessage

from agents.graph.analyst_graph import create_synthesizer_node
from agents.planner import QueryPlan, AnalysisStep


def _step_result(tool: str, raw: str):
    """Build a StepResult dict mimicking a prose-returning tool."""
    return {
        "tool": tool,
        "data": {},
        "raw": raw,
        "filing_ref": None,
        "error": None,
    }


def _make_state(query, plan, step_results):
    """Build a minimal state dict for the synthesizer."""
    return {
        "messages": [HumanMessage(content=query)],
        "ticker": "AAPL",
        "query_complexity": "complex",
        "classification": None,
        "plan": plan,
        "step_results": step_results,
        "final_response": "",
    }


@pytest.mark.eval_fast
class TestSynthesizer:
    """Test create_synthesizer_node with mock step results."""

    async def test_produces_nonempty_response(self, llm):
        """Synthesizer should produce a meaningful response, not empty string."""
        plan = QueryPlan(
            query_type="complex",
            requires_planning=True,
            steps=[
                AnalysisStep(
                    id=1,
                    action="Get stock info",
                    tool="get_stock_info",
                    rationale="basic data",
                ),
            ],
            synthesis_approach="Summarize the findings",
        )
        step_results = {
            1: _step_result(
                "get_stock_info",
                "Stock Info for AAPL: Current Price $185.50, P/E Ratio 28.5, Market Cap $2.85T",
            )
        }
        state = _make_state("What is Apple's current stock info?", plan, step_results)

        synthesizer = create_synthesizer_node(llm, "AAPL")
        result_state = await synthesizer(state)

        assert result_state["final_response"]
        assert len(result_state["final_response"]) > 20

    async def test_synthesizer_surfaces_conflicts(self, llm):
        """When the reconciler flags a cross-tool conflict, the synthesizer must
        acknowledge the disagreement instead of emitting two contradictory
        descriptions of the same filing as separate sections.

        Reproduces the GOOGL bug shape from #30: two tools describing the same
        8-K filing but disagreeing on event_type.
        """
        plan = QueryPlan(
            query_type="complex",
            requires_planning=True,
            steps=[
                AnalysisStep(
                    id=1, action="Overview of latest 8-K", tool="get_8k_overview", rationale="overview",
                ),
                AnalysisStep(
                    id=2, action="Material event summary", tool="get_material_event_summary", rationale="event",
                ),
            ],
            synthesis_approach="Describe the latest 8-K filing",
        )
        step_results = {
            1: _step_result(
                "get_8k_overview",
                "Filing accession 0000320193-25-000001 dated 2025-04-01: earnings release for Q1.",
            ),
            2: _step_result(
                "get_material_event_summary",
                "Filing accession 0000320193-25-000001 dated 2025-04-01: management guidance update.",
            ),
        }
        state = _make_state("Describe Apple's latest 8-K", plan, step_results)
        # Inject a synthetic conflict as the reconciler would emit
        state["conflicts"] = [
            {
                "filing_ref": "0000320193-25-000001",
                "step_ids": [1, 2],
                "field": "event_type",
                "values": ["earnings_release", "guidance_update"],
            }
        ]

        synthesizer = create_synthesizer_node(llm, "AAPL")
        result_state = await synthesizer(state)
        response = result_state["final_response"].lower()

        # Response should mention BOTH event types (acknowledging the conflict),
        # not pick one and ignore the other.
        assert any(t in response for t in ["earnings", "release"]), (
            f"Response missed earnings_release: {response[:300]}"
        )
        assert any(t in response for t in ["guidance", "guidance_update"]), (
            f"Response missed guidance_update: {response[:300]}"
        )

    async def test_references_all_step_results(self, llm):
        """Synthesizer must reference content from ALL steps, not just the last.

        We give it two distinct mock results (risk analysis + stock price) and
        check that the final response mentions content from both. This guards
        against the recency bias failure mode.
        """
        plan = QueryPlan(
            query_type="complex",
            requires_planning=True,
            steps=[
                AnalysisStep(
                    id=1,
                    action="Analyze risks",
                    tool="get_risk_factors_summary",
                    rationale="identify risks",
                ),
                AnalysisStep(
                    id=2,
                    action="Get stock price",
                    tool="get_stock_price_history",
                    rationale="current pricing",
                ),
            ],
            synthesis_approach="Combine risk analysis with price context",
        )
        step_results = {
            1: _step_result(
                "get_risk_factors_summary",
                "Risk Analysis: Apple faces supply chain concentration risk in China and regulatory antitrust pressure in the EU.",
            ),
            2: _step_result(
                "get_stock_price_history",
                "Stock Price: AAPL closed at $187.50 on 2025-01-15, up 2.3% over the past month.",
            ),
        }
        state = _make_state(
            "Analyze Apple's risks and current stock performance", plan, step_results
        )

        synthesizer = create_synthesizer_node(llm, "AAPL")
        result_state = await synthesizer(state)

        response = result_state["final_response"].lower()

        # Should mention risk-related content from step 1
        assert any(
            term in response for term in ["risk", "supply chain", "regulatory", "antitrust"]
        ), f"Response doesn't reference step 1 (risk analysis): {response[:300]}"

        # Should mention price-related content from step 2
        assert any(
            term in response for term in ["price", "$187", "187.50", "stock"]
        ), f"Response doesn't reference step 2 (stock price): {response[:300]}"
