"""Tests for the step executor node (create_step_executor_node).

Marker: eval_tools — requires API keys because tools hit SEC/Yahoo APIs.

The step executor is a factory that returns a function. That function:
1. Reads the current step from state["plan"].steps[state["current_step_index"]]
2. Calls tools_dict[step.tool]("") to execute it
3. Stores the result in state["step_results"][step.id]
4. Increments current_step_index

We test it by constructing state dicts and calling the node function directly,
bypassing the LangGraph runtime entirely. This isolates the executor logic.
"""

import pytest

from agents.graph.sec_graph import create_step_executor_node
from agents.planner import QueryPlan, AnalysisStep


def _make_state(plan, step_index=0):
    """Build a minimal state dict for the step executor."""
    return {
        "messages": [],
        "ticker": "AAPL",
        "query_complexity": "complex",
        "classification": None,
        "plan": plan,
        "current_step_index": step_index,
        "step_results": {},
        "final_response": "",
    }


def _make_plan(tool_names):
    """Create a QueryPlan with steps using the given tool names."""
    steps = [
        AnalysisStep(
            id=i + 1,
            action=f"Execute {name}",
            tool=name,
            rationale="test",
        )
        for i, name in enumerate(tool_names)
    ]
    return QueryPlan(
        query_type="complex",
        requires_planning=True,
        steps=steps,
        synthesis_approach="combine",
    )


@pytest.mark.eval_tools
class TestStepExecutor:
    """Test create_step_executor_node by calling the returned function directly."""

    def test_executes_valid_tool(self, tools_dict):
        """Executor should call the tool and store a non-empty result.

        get_stock_info is a fast, reliable tool (Yahoo Finance) that doesn't
        require SEC filings to load, making it ideal for this test.
        """
        plan = _make_plan(["get_stock_info"])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "AAPL")
        result_state = executor(state)

        # Step result was stored
        assert 1 in result_state["step_results"]
        result = result_state["step_results"][1]
        assert len(result) > 50, f"Result too short: {result[:100]}"
        assert "[ERROR" not in result

        # Index advanced
        assert result_state["current_step_index"] == 1

    def test_handles_invalid_tool_gracefully(self, tools_dict):
        """An invalid tool name should produce an error string, not crash.

        The executor stores an [ERROR: ...] message in step_results rather
        than raising an exception, because the synthesizer can still produce
        a partial answer from other steps.
        """
        plan = _make_plan(["nonexistent_tool_xyz"])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "AAPL")
        result_state = executor(state)

        assert 1 in result_state["step_results"]
        assert "[ERROR" in result_state["step_results"][1]
        assert result_state["current_step_index"] == 1

    def test_multi_step_sequential(self, tools_dict):
        """Executor processes one step per call, advancing the index each time.

        This simulates the LangGraph loop: the graph calls step_executor
        repeatedly until check_more_steps routes to synthesizer.
        """
        plan = _make_plan(["get_stock_info", "get_stock_price_history"])
        state = _make_state(plan)

        executor = create_step_executor_node(tools_dict, "AAPL")

        # First call — executes step 1
        state = executor(state)
        assert state["current_step_index"] == 1
        assert 1 in state["step_results"]

        # Second call — executes step 2
        state = executor(state)
        assert state["current_step_index"] == 2
        assert 2 in state["step_results"]
