"""Tests for streaming functionality in sec_graph.py.

Marker: eval_unit — NO API keys needed. These test the streaming logic
by mocking the LLM and workflow, verifying that stream_sync() yields
the correct events in the correct order.

Tests cover:
- _process_streaming_chunk: parsing string vs list content from LLM chunks
- stream_sync: node events, tool events, response events from graph streaming
- Backward compat: invoke() still works after the refactor
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage

from agents.graph.sec_graph import (
    _process_streaming_chunk,
    _extract_text_content,
    PlanningAgent,
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


class TestExtractTextContent:
    """_extract_text_content extracts text from Gemini's response formats."""

    @pytest.mark.eval_unit
    def test_string_passthrough(self):
        assert _extract_text_content("hello world") == "hello world"

    @pytest.mark.eval_unit
    def test_list_with_text_block(self):
        content = [{"type": "text", "text": "the answer"}]
        assert _extract_text_content(content) == "the answer"

    @pytest.mark.eval_unit
    def test_list_with_thinking_and_text(self):
        """Should extract only text, ignoring thinking blocks."""
        content = [
            {"type": "thinking", "thinking": "let me reason..."},
            {"type": "text", "text": "final answer"},
        ]
        assert _extract_text_content(content) == "final answer"

    @pytest.mark.eval_unit
    def test_list_with_raw_string(self):
        content = ["raw string part"]
        assert _extract_text_content(content) == "raw string part"

    @pytest.mark.eval_unit
    def test_empty_string(self):
        assert _extract_text_content("") == ""


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
        """A complex query should yield tool events from step_executor."""
        plan = _make_plan(2)

        mock_workflow = MagicMock()
        mock_workflow.stream.return_value = [
            ("updates", {"router": {"query_complexity": "complex"}}),
            ("updates", {"planner": {"plan": plan, "current_step_index": 0}}),
            ("updates", {"step_executor": {"current_step_index": 1, "step_results": {1: "result1"}}}),
            ("updates", {"step_executor": {"current_step_index": 2, "step_results": {2: "result2"}}}),
            ("updates", {"synthesizer": {"final_response": "Combined analysis..."}}),
        ]

        agent = PlanningAgent(mock_workflow, "AAPL")
        events = list(agent.stream_sync({"messages": [HumanMessage(content="analyze risks")]}))

        # Should have tool events for each step
        tool_events = [e for e in events if e["type"] == "tool"]
        assert len(tool_events) == 2
        assert tool_events[0]["tool"] == "tool_1"
        assert tool_events[0]["step"] == 1
        assert tool_events[0]["total"] == 2
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
