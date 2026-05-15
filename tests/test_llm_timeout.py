"""LLM-call timeout tests.

Every llm.invoke / ainvoke / astream site in the analyst graph is wrapped in
`asyncio.wait_for(..., timeout=LLM_CALL_TIMEOUT)`. A hung LLM (Gemini 3 with
thinking can hang indefinitely on adversarial prompts) must not pin a WS
session in memory — it should surface as a clean error.
"""

import asyncio

import pytest

from agents.graph import analyst_graph
from agents.graph.analyst_graph import (
    LLMTimeoutError,
    _run_with_timeout,
    _truncate_messages,
)
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_run_with_timeout_returns_result_under_deadline():
    async def fast():
        return "ok"

    assert await _run_with_timeout(fast(), label="t") == "ok"


@pytest.mark.asyncio
async def test_run_with_timeout_raises_llmtimeouterror(monkeypatch):
    monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)

    async def slow():
        await asyncio.sleep(1.0)
        return "never"

    with pytest.raises(LLMTimeoutError, match="timed out"):
        await _run_with_timeout(slow(), label="test_label")


@pytest.mark.asyncio
async def test_run_with_timeout_includes_label_in_error(monkeypatch):
    monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)

    async def slow():
        await asyncio.sleep(0.5)

    with pytest.raises(LLMTimeoutError, match="my_label"):
        await _run_with_timeout(slow(), label="my_label")


# ---------------------------------------------------------------------------
# Node-level integration: router_node propagates LLMTimeoutError
# ---------------------------------------------------------------------------


class _FakePlanner:
    """Minimal QueryPlanner stand-in. classify_query/create_plan are sync (the
    real planner calls llm.invoke); these stubs sleep to trigger the timeout."""

    def __init__(self, sleep_seconds: float):
        self._sleep = sleep_seconds

    def classify_query(self, query):
        import time
        time.sleep(self._sleep)
        return None

    def create_plan(self, query):
        import time
        time.sleep(self._sleep)
        return None


@pytest.mark.asyncio
async def test_router_node_times_out(monkeypatch):
    monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)

    router = analyst_graph.create_router_node(_FakePlanner(sleep_seconds=0.5))
    state = {"messages": [HumanMessage(content="hello")], "ticker": "AAPL"}

    with pytest.raises(LLMTimeoutError):
        await router(state)


@pytest.mark.asyncio
async def test_planner_node_times_out(monkeypatch):
    monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)

    planner_node = analyst_graph.create_planner_node(_FakePlanner(sleep_seconds=0.5))
    state = {"messages": [HumanMessage(content="hello")], "ticker": "AAPL"}

    with pytest.raises(LLMTimeoutError):
        await planner_node(state)


# ---------------------------------------------------------------------------
# History truncation
# ---------------------------------------------------------------------------


def test_truncate_messages_under_limit():
    msgs = [HumanMessage(content=f"q{i}") for i in range(5)]
    assert _truncate_messages(msgs, limit=30) is msgs


def test_truncate_messages_over_limit_keeps_latest():
    msgs = []
    for i in range(60):
        msgs.append(HumanMessage(content=f"q{i}"))
        msgs.append(AIMessage(content=f"a{i}"))

    out = _truncate_messages(msgs, limit=30)
    assert len(out) == 30
    # Latest user message must be preserved (it's at the tail)
    assert out[-1].content == "a59"
    assert out[-2].content == "q59"


def test_truncate_messages_default_limit():
    msgs = [HumanMessage(content=str(i)) for i in range(100)]
    out = _truncate_messages(msgs)
    assert len(out) == analyst_graph.MESSAGES_HISTORY_LIMIT
