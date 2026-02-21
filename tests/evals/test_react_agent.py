"""End-to-end tests for the simple query path (router → ReAct agent).

Marker: eval_e2e — full agent invocations that exercise the entire graph.

These test the "happy path" for simple queries: the router classifies them
as simple, routes to the ReAct agent, which picks the right tool and
returns a formatted answer. Each test is a full graph invocation.
"""

import pytest

from langchain_core.messages import HumanMessage


@pytest.mark.eval_e2e
class TestReactAgent:
    """Test simple query end-to-end through the full PlanningAgent."""

    def test_stock_price_query(self, agent):
        """A stock price query should return a response with price data.

        The ReAct agent should pick get_stock_info or get_stock_price_history
        and format a response containing a dollar sign or the word 'price'.
        """
        result = agent.invoke({
            "messages": [HumanMessage(content="What is Apple's current stock price?")]
        })

        assert "messages" in result
        response = result["messages"][-1].content.lower()
        assert "$" in response or "price" in response, (
            f"Response doesn't mention price: {response[:300]}"
        )

    def test_pe_ratio_query(self, agent):
        """A P/E ratio query should mention 'p/e' in the response."""
        result = agent.invoke({
            "messages": [HumanMessage(content="What is AAPL's P/E ratio?")]
        })

        assert "messages" in result
        response = result["messages"][-1].content.lower()
        assert "p/e" in response or "pe" in response or "price" in response, (
            f"Response doesn't mention P/E: {response[:300]}"
        )

    def test_risk_factors_query(self, agent):
        """A risk factors query should return substantial content (>200 chars).

        Risk factors are complex SEC filing data. A short or generic response
        (like 'Apple has some risks') indicates the agent didn't call the
        tool or the tool failed silently.
        """
        result = agent.invoke({
            "messages": [HumanMessage(content="What are the main risk factors for Apple?")]
        })

        assert "messages" in result
        response = result["messages"][-1].content
        assert len(response) > 200, (
            f"Risk factors response too short ({len(response)} chars), "
            f"likely generic: {response[:200]}"
        )
