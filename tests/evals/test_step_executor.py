"""Tests for the worker node (create_worker_node).

Marker: eval_tools — requires API keys because tools hit SEC/Yahoo APIs.

The worker is what each `Send` from `dispatch_steps` lands on. It receives
`{"step": AnalysisStep}`, runs `tools_dict[step.tool].invoke("")`, and
returns `{"step_results": {step.id: StepResult}}` for the reducer to merge.

We test it by calling the node function directly with a synthetic Send
payload — this isolates the worker logic from the LangGraph runtime.
"""

import pytest

from agents.graph.analyst_graph import create_worker_node
from agents.planner import AnalysisStep


def _make_step(step_id: int, tool_name: str) -> AnalysisStep:
    return AnalysisStep(
        id=step_id,
        action=f"Execute {tool_name}",
        tool=tool_name,
        rationale="test",
    )


@pytest.mark.eval_tools
class TestWorker:
    """Call the worker function directly with a synthetic Send payload."""

    def test_executes_valid_tool(self, tools_dict):
        """Worker should call the tool and produce a non-empty StepResult.

        get_stock_info is a fast, reliable Yahoo Finance tool — no SEC load.
        """
        worker = create_worker_node(tools_dict)
        delta = worker({"step": _make_step(1, "get_stock_info")})

        assert 1 in delta["step_results"]
        result = delta["step_results"][1]
        assert result["error"] is None
        assert len(result["raw"]) > 50, f"Result too short: {result['raw'][:100]}"
        assert "[ERROR" not in result["raw"]

    def test_handles_invalid_tool_gracefully(self, tools_dict):
        """An invalid tool name should produce an error StepResult, not crash."""
        worker = create_worker_node(tools_dict)
        delta = worker({"step": _make_step(1, "nonexistent_tool_xyz")})

        result = delta["step_results"][1]
        assert "[ERROR" in result["raw"]
        assert result["error"] is not None
