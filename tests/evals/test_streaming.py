"""Tests for streaming functionality in analyst_graph.py.

Marker: eval_unit — NO API keys needed. These test the streaming logic
by mocking the LLM and workflow, verifying that stream_sync() yields
the correct events in the correct order.

Tests cover:
- _process_streaming_chunk: parsing string vs list content from LLM chunks
- stream_sync: node events, tool events, response events from graph streaming
- Per-step streaming: workers emit one tool event each via Send fan-out
- Backward compat: invoke() still works after the refactor
"""

import time

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool

from agents.graph.analyst_graph import (
    _process_streaming_chunk,
    PlanningAgent,
    AnalysisState,
    create_worker_node,
    create_reconciler_node,
    dispatch_steps,
)
from agents.planner import QueryPlan, AnalysisStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeChunk:
    """Mimics an AIMessageChunk with a content field.

    Gemini returns two content formats:
    - str: regular text token (standard streaming)
    - list[dict]: structured blocks when include_thoughts=True
    """

    content: object  # str | list[dict]


def _make_plan(num_steps: int) -> QueryPlan:
    """Create a QueryPlan with num_steps dummy steps."""
    steps = [
        AnalysisStep(
            id=i,
            action=f"Step {i} action",
            tool=f"tool_{i}",
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
# _process_streaming_chunk tests
# ---------------------------------------------------------------------------


class TestProcessStreamingChunk:
    """_process_streaming_chunk parses LLM chunks and emits events via writer."""

    @pytest.mark.eval_unit
    def test_string_content_emits_token(self):
        """A chunk with string content should emit a token event."""
        chunk = FakeChunk(content="Hello")
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        assert text == "Hello"
        assert len(events) == 1
        assert events[0] == {"type": "token", "message": "Hello"}

    @pytest.mark.eval_unit
    def test_empty_string_emits_nothing(self):
        """Empty string content should not emit any event."""
        chunk = FakeChunk(content="")
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        assert text == ""
        assert len(events) == 0

    @pytest.mark.eval_unit
    def test_thinking_block_emits_thinking(self):
        """A list with a thinking block should emit a thinking event."""
        chunk = FakeChunk(content=[
            {"type": "thinking", "thinking": "Let me analyze..."}
        ])
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        # Thinking blocks don't contribute to text content
        assert text == ""
        assert len(events) == 1
        assert events[0] == {"type": "thinking", "message": "Let me analyze..."}

    @pytest.mark.eval_unit
    def test_reasoning_block_emits_thinking(self):
        """A list with a 'reasoning' block (v1 format) should also emit thinking."""
        chunk = FakeChunk(content=[
            {"type": "reasoning", "reasoning": "Step by step..."}
        ])
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        assert text == ""
        assert events[0] == {"type": "thinking", "message": "Step by step..."}

    @pytest.mark.eval_unit
    def test_text_block_emits_token(self):
        """A list with a text block should emit a token event."""
        chunk = FakeChunk(content=[
            {"type": "text", "text": "Apple's revenue"}
        ])
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        assert text == "Apple's revenue"
        assert events[0] == {"type": "token", "message": "Apple's revenue"}

    @pytest.mark.eval_unit
    def test_mixed_blocks(self):
        """A list with both thinking and text blocks should emit both event types."""
        chunk = FakeChunk(content=[
            {"type": "thinking", "thinking": "Analyzing risk factors..."},
            {"type": "text", "text": "Based on my analysis"},
        ])
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        assert text == "Based on my analysis"
        assert len(events) == 2
        assert events[0]["type"] == "thinking"
        assert events[1]["type"] == "token"

    @pytest.mark.eval_unit
    def test_non_dict_list_items_skipped(self):
        """Non-dict items in a list content should be silently skipped."""
        chunk = FakeChunk(content=["raw string", 42, {"type": "text", "text": "ok"}])
        events = []
        writer = lambda e: events.append(e)

        text = _process_streaming_chunk(chunk, writer)

        assert text == "ok"
        assert len(events) == 1


# ---------------------------------------------------------------------------
# PlanningAgent.stream_sync tests
# ---------------------------------------------------------------------------


class TestStreamSync:
    """stream_sync() yields structured events from the graph execution."""

    @pytest.mark.eval_unit
    def test_simple_query_yields_node_and_response(self):
        """A simple query should yield router + react_agent node events and a response."""
        # Mock workflow.stream to return updates for a simple path:
        # router → react_agent
        mock_workflow = MagicMock()
        mock_workflow.stream.return_value = [
            ("updates", {"router": {"query_complexity": "simple"}}),
            ("updates", {"react_agent": {"final_response": "AAPL is at $185"}}),
        ]

        agent = PlanningAgent(mock_workflow, "AAPL")
        events = list(agent.stream_sync({"messages": [HumanMessage(content="price?")]}))

        # Should have: router node, react_agent node, response
        types = [e["type"] for e in events]
        assert "node" in types
        assert "response" in types

        # Response should contain the answer
        response_events = [e for e in events if e["type"] == "response"]
        assert response_events[0]["message"] == "AAPL is at $185"

    @pytest.mark.eval_unit
    def test_complex_query_yields_tool_events(self):
        """A complex query should yield one tool event per worker emission.

        With Send-based fan-out, each worker is its own LangGraph node
        invocation, so `astream(stream_mode="updates")` emits one chunk per
        worker as it completes. Each chunk's step_results delta carries the
        single step_id that worker just wrote.
        """
        plan = _make_plan(2)

        mock_workflow = MagicMock()
        mock_workflow.stream.return_value = [
            ("updates", {"router": {"query_complexity": "complex"}}),
            ("updates", {"planner": {"plan": plan}}),
            ("updates", {"worker": {"step_results": {1: {"tool": "tool_1", "raw": "r1", "data": {}, "filing_ref": None, "error": None}}}}),
            ("updates", {"worker": {"step_results": {2: {"tool": "tool_2", "raw": "r2", "data": {}, "filing_ref": None, "error": None}}}}),
            ("updates", {"synthesizer": {"final_response": "Combined analysis..."}}),
        ]

        agent = PlanningAgent(mock_workflow, "AAPL")
        events = list(agent.stream_sync({"messages": [HumanMessage(content="analyze risks")]}))

        tool_events = [e for e in events if e["type"] == "tool"]
        assert len(tool_events) == 2
        assert tool_events[0]["tool"] == "tool_1"
        assert tool_events[0]["step"] == 1
        assert tool_events[0]["total"] == 2
        assert tool_events[0]["node"] == "worker"
        assert tool_events[1]["tool"] == "tool_2"
        assert tool_events[1]["step"] == 2

    @pytest.mark.eval_unit
    def test_custom_events_passed_through(self):
        """Custom events from get_stream_writer (thinking/tokens) should be yielded as-is."""
        mock_workflow = MagicMock()
        mock_workflow.stream.return_value = [
            ("updates", {"router": {"query_complexity": "simple"}}),
            ("custom", {"type": "thinking", "message": "Let me think..."}),
            ("custom", {"type": "token", "message": "Apple"}),
            ("updates", {"react_agent": {"final_response": "Apple analysis"}}),
        ]

        agent = PlanningAgent(mock_workflow, "AAPL")
        events = list(agent.stream_sync({"messages": [HumanMessage(content="analyze")]}))

        thinking = [e for e in events if e["type"] == "thinking"]
        tokens = [e for e in events if e["type"] == "token"]

        assert len(thinking) == 1
        assert thinking[0]["message"] == "Let me think..."
        assert len(tokens) == 1
        assert tokens[0]["message"] == "Apple"

    @pytest.mark.eval_unit
    def test_node_messages_are_descriptive(self):
        """Node events should include human-readable status messages."""
        mock_workflow = MagicMock()
        mock_workflow.stream.return_value = [
            ("updates", {"router": {"query_complexity": "simple"}}),
            ("updates", {"react_agent": {"final_response": "done"}}),
        ]

        agent = PlanningAgent(mock_workflow, "AAPL")
        events = list(agent.stream_sync({"messages": [HumanMessage(content="test")]}))

        node_events = [e for e in events if e["type"] == "node"]
        assert any("Classifying" in e["message"] for e in node_events)
        assert any("Processing" in e["message"] for e in node_events)


# ---------------------------------------------------------------------------
# Per-step streaming via real Send fan-out
# ---------------------------------------------------------------------------


def _make_streaming_workflow(tools_dict, plan):
    """Compile a minimal StateGraph with the real Send fan-out, so
    stream_sync() exercises actual LangGraph streaming semantics."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AnalysisState)
    # Stub planner just injects the pre-built plan into state
    graph.add_node("planner_stub", lambda state: {"plan": plan, "step_results": {}})
    graph.add_node("worker", create_worker_node(tools_dict))
    graph.add_node("reconciler", create_reconciler_node())
    graph.set_entry_point("planner_stub")
    graph.add_conditional_edges(
        "planner_stub", dispatch_steps, ["worker", END]
    )
    graph.add_edge("worker", "reconciler")
    graph.add_edge("reconciler", END)
    return graph.compile()


class TestPerStepStreaming:
    """Send-based fan-out emits one updates chunk per worker as it completes,
    so stream_sync() should yield one tool event per step."""

    @pytest.mark.eval_unit
    def test_worker_events_stream_per_step(self):
        """N-step plan → N tool events, one per worker emission."""
        tools_dict = {
            "tool_a": Tool.from_function(name="tool_a", description="a", func=lambda q="": "ra"),
            "tool_b": Tool.from_function(name="tool_b", description="b", func=lambda q="": "rb"),
            "tool_c": Tool.from_function(name="tool_c", description="c", func=lambda q="": "rc"),
        }
        plan = QueryPlan(
            query_type="complex",
            requires_planning=True,
            steps=[
                AnalysisStep(id=1, action="A", tool="tool_a", rationale="r"),
                AnalysisStep(id=2, action="B", tool="tool_b", rationale="r"),
                AnalysisStep(id=3, action="C", tool="tool_c", rationale="r"),
            ],
            synthesis_approach="combine",
        )
        workflow = _make_streaming_workflow(tools_dict, plan)
        agent = PlanningAgent(workflow, "TEST")

        events = list(agent.stream_sync({"messages": [HumanMessage(content="q")]}))

        tool_events = [e for e in events if e["type"] == "tool"]
        assert len(tool_events) == 3, f"Expected 3 tool events, got {len(tool_events)}: {tool_events}"
        assert {e["step"] for e in tool_events} == {1, 2, 3}
        assert all(e["total"] == 3 for e in tool_events)
        assert all(e["node"] == "worker" for e in tool_events)

    @pytest.mark.eval_unit
    def test_worker_events_carry_correct_step_metadata(self):
        """Each tool event carries the right tool/action for its step."""
        tools_dict = {
            "tool_a": Tool.from_function(name="tool_a", description="a", func=lambda q="": "ra"),
            "tool_b": Tool.from_function(name="tool_b", description="b", func=lambda q="": "rb"),
        }
        plan = QueryPlan(
            query_type="complex",
            requires_planning=True,
            steps=[
                AnalysisStep(id=1, action="Analyze risks", tool="tool_a", rationale="r"),
                AnalysisStep(id=2, action="Get prices", tool="tool_b", rationale="r"),
            ],
            synthesis_approach="combine",
        )
        workflow = _make_streaming_workflow(tools_dict, plan)
        agent = PlanningAgent(workflow, "TEST")

        events = list(agent.stream_sync({"messages": [HumanMessage(content="q")]}))
        tool_events = sorted(
            (e for e in events if e["type"] == "tool"), key=lambda e: e["step"]
        )

        assert tool_events[0]["tool"] == "tool_a"
        assert tool_events[0]["action"] == "Analyze risks"
        assert tool_events[1]["tool"] == "tool_b"
        assert tool_events[1]["action"] == "Get prices"

    @pytest.mark.eval_unit
    def test_worker_events_arrive_as_workers_complete(self):
        """Workers running at staggered speeds emit their tool events at
        staggered times — the user sees per-step UX, not a batch after the
        slowest worker."""
        slow_finished_at: dict[str, float] = {}
        first_event_at: list[float] = []

        def slow(name: str, delay: float):
            def fn(q=""):
                time.sleep(delay)
                slow_finished_at[name] = time.monotonic()
                return f"done {name}"
            return Tool.from_function(name=name, description=name, func=fn)

        tools_dict = {
            "fast_a": slow("fast_a", 0.05),
            "slow_b": slow("slow_b", 0.30),
        }
        plan = QueryPlan(
            query_type="complex",
            requires_planning=True,
            steps=[
                AnalysisStep(id=1, action="A", tool="fast_a", rationale="r"),
                AnalysisStep(id=2, action="B", tool="slow_b", rationale="r"),
            ],
            synthesis_approach="combine",
        )
        workflow = _make_streaming_workflow(tools_dict, plan)
        agent = PlanningAgent(workflow, "TEST")

        events = []
        start = time.monotonic()
        for ev in agent.stream_sync({"messages": [HumanMessage(content="q")]}):
            events.append((time.monotonic() - start, ev))
            if ev.get("type") == "tool" and not first_event_at:
                first_event_at.append(time.monotonic() - start)

        tool_events = [(t, e) for t, e in events if e["type"] == "tool"]
        assert len(tool_events) == 2

        # The first tool event should arrive well before the slow worker finishes,
        # proving per-step emission rather than batched-after-everything.
        assert first_event_at[0] < 0.25, (
            f"First tool event arrived at {first_event_at[0]:.2f}s — looks batched "
            f"(slow worker takes 0.30s)"
        )


# ---------------------------------------------------------------------------
# Backward compatibility: invoke() still works
# ---------------------------------------------------------------------------


class TestInvokeBackwardCompat:
    """invoke() should work exactly as before the streaming refactor."""

    @pytest.mark.eval_unit
    def test_invoke_returns_messages_dict(self):
        """invoke() should return {"messages": [...]} with the response appended."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "final_response": "AAPL is trading at $185",
            "messages": [HumanMessage(content="price?")],
        }

        agent = PlanningAgent(mock_workflow, "AAPL")
        result = agent.invoke({"messages": [HumanMessage(content="price?")]})

        assert "messages" in result
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "AAPL is trading at $185"

    @pytest.mark.eval_unit
    def test_invoke_with_empty_response(self):
        """invoke() with no final_response should return messages unchanged."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "final_response": "",
            "messages": [HumanMessage(content="test")],
        }

        agent = PlanningAgent(mock_workflow, "AAPL")
        result = agent.invoke({"messages": [HumanMessage(content="test")]})

        assert "messages" in result
        assert len(result["messages"]) == 1  # No AIMessage added
