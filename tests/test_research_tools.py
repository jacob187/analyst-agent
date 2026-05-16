"""Verifies the Tavily deep-research tool releases the event loop between polls.

Pre-fix: `_tool_tavily_research` slept the worker thread for `time.sleep(2)` ×
up to 30 iterations (60s). Because LangGraph offloads the sync `worker` node
to the asyncio thread pool, each in-flight deep_research call pinned one of
the ~13 pool threads for the entire poll, starving other `asyncio.to_thread`
work (charts, briefing, SEC tools).

Post-fix: the async variant uses `await asyncio.sleep(...)` between polls and
wraps `client.research` / `client.get_research` in `asyncio.to_thread`, so the
loop and the thread pool both stay free.
"""

import asyncio

import pytest

from agents.tools import research_tools


class _FakeTavilyClient:
    """Returns in_progress twice, then completed. No real network."""

    def __init__(self, api_key: str):  # noqa: ARG002 — matches TavilyClient signature
        self._calls = 0

    def research(self, input: str):  # noqa: A002 — Tavily SDK uses `input`
        return {"request_id": "req_test_123"}

    def get_research(self, request_id: str):
        self._calls += 1
        if self._calls < 3:
            return {"status": "in_progress"}
        return {
            "status": "completed",
            "content": "Synthesized research body.",
            "sources": [{"title": "S1", "url": "https://example.com/1"}],
        }


@pytest.fixture(autouse=True)
def _clear_research_cache():
    research_tools._research_cache.clear()
    yield
    research_tools._research_cache.clear()


@pytest.fixture
def _fast_polling(monkeypatch):
    """Shorten the poll interval so tests don't wait the production 2s default."""
    monkeypatch.setattr(research_tools, "_RESEARCH_POLL_INTERVAL_SECONDS", 0.05)


@pytest.mark.asyncio
async def test_tavily_research_async_returns_completed_content(
    monkeypatch, _fast_polling
):
    monkeypatch.setattr(research_tools, "TavilyClient", _FakeTavilyClient)

    out = await research_tools._tool_tavily_research_async(
        "AAPL", "supply chain", "fake-key"
    )

    assert "Synthesized research body." in out
    assert "Deep Research Report: AAPL - supply chain" in out
    assert "S1" in out


@pytest.mark.asyncio
async def test_tavily_research_does_not_block_loop(monkeypatch, _fast_polling):
    """While the tool polls, a concurrent ticker must continue advancing."""
    monkeypatch.setattr(research_tools, "TavilyClient", _FakeTavilyClient)

    counter = 0

    async def tick():
        nonlocal counter
        while True:
            await asyncio.sleep(0.01)
            counter += 1

    ticker_task = asyncio.create_task(tick())
    try:
        await research_tools._tool_tavily_research_async(
            "AAPL", "supply chain", "fake-key"
        )
    finally:
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass

    # Two polls × 50ms = 100ms minimum, ticker fires every 10ms → expect ≥ 5.
    # Generous lower bound to tolerate scheduler jitter.
    assert counter >= 5, (
        f"loop blocked during Tavily polling — ticker advanced only {counter} times"
    )


@pytest.mark.asyncio
async def test_tavily_research_returns_cached_without_polling(monkeypatch):
    """Cache hit must not call TavilyClient at all."""
    research_tools._research_cache[research_tools._get_cache_key("AAPL", "x")] = {
        "content": "cached body",
        "sources": 1,
    }

    def boom(*args, **kwargs):
        raise AssertionError("Tavily must not be called on cache hit")

    monkeypatch.setattr(research_tools, "TavilyClient", boom)

    out = await research_tools._tool_tavily_research_async("AAPL", "x", "fake-key")
    assert "[Cached Research]" in out
    assert "cached body" in out


@pytest.mark.asyncio
async def test_tavily_research_failed_status(monkeypatch, _fast_polling):
    class _Failing(_FakeTavilyClient):
        def get_research(self, request_id):
            return {"status": "failed", "error": "quota_exceeded"}

    monkeypatch.setattr(research_tools, "TavilyClient", _Failing)

    out = await research_tools._tool_tavily_research_async("AAPL", "x", "fake-key")
    assert "Deep research failed" in out
    assert "quota_exceeded" in out


@pytest.mark.asyncio
async def test_tavily_research_no_request_id(monkeypatch, _fast_polling):
    class _NoRequestId:
        def __init__(self, api_key):
            pass

        def research(self, input):
            return {}

    monkeypatch.setattr(research_tools, "TavilyClient", _NoRequestId)

    out = await research_tools._tool_tavily_research_async("AAPL", "x", "fake-key")
    assert "no request_id" in out


@pytest.mark.asyncio
async def test_deep_research_tool_dispatches_async_coroutine(monkeypatch, _fast_polling):
    """The `deep_research` Tool must invoke the async coroutine on .ainvoke."""
    monkeypatch.setattr(research_tools, "TavilyClient", _FakeTavilyClient)

    tools = research_tools.create_research_tools("AAPL", "fake-key")
    deep_research = next(t for t in tools if t.name == "deep_research")

    # ainvoke routes through the registered coroutine; if it fell back to the
    # sync `func`, the inner asyncio.run would error because we're already in a
    # running loop.
    result = await deep_research.ainvoke("supply chain")
    assert "Synthesized research body." in result
