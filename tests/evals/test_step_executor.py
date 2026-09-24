"""Tests for the worker node (create_worker_node).

Marker: eval_tools — requires API keys because tools hit SEC/Yahoo APIs.

The worker is what each `Send` from `dispatch_steps` lands on. It receives
`{"step": AnalysisStep}`, awaits `tools_dict[step.tool].ainvoke("")`, and
returns `{"step_results": {step.id: StepResult}}` for the reducer to merge.

We test it by awaiting the node function directly with a synthetic Send
payload — this isolates the worker logic from the LangGraph runtime.
"""

import pytest

from agents.graph.analyst_graph import _build_tools_dict, create_worker_node
from agents.planner import AnalysisStep


@pytest.fixture
def worker_tools(tools):
    """Name → Tool, built the way the graph builds it.

    The shared `tools_dict` fixture maps to bound `.invoke` methods for the
    convenience of tests that call `tools_dict[name]("")`; the worker awaits
    `.ainvoke` on the Tool itself.
    """
    return _build_tools_dict(tools)


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

    async def test_executes_valid_tool(self, worker_tools):
        """Worker should call the tool and produce a non-empty StepResult.

        get_stock_info is a fast, reliable Yahoo Finance tool — no SEC load.
        """
        worker = create_worker_node(worker_tools)
        delta = await worker({"step": _make_step(1, "get_stock_info")})

        assert 1 in delta["step_results"]
        result = delta["step_results"][1]
        assert result["error"] is None
        assert len(result["raw"]) > 50, f"Result too short: {result['raw'][:100]}"
        assert "[ERROR" not in result["raw"]

    async def test_handles_invalid_tool_gracefully(self, worker_tools):
        """An invalid tool name should produce an error StepResult, not crash."""
        worker = create_worker_node(worker_tools)
        delta = await worker({"step": _make_step(1, "nonexistent_tool_xyz")})

        result = delta["step_results"][1]
        assert "[ERROR" in result["raw"]
        assert result["error"] is not None
