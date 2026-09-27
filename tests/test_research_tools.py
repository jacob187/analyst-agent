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
from langchain_core.tools import ToolException

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


# ── Free-trial metering ──────────────────────────────────────────────────────


class _RecordingTavilySearch:
    """Stands in for TavilySearch at the network boundary: records constructor
    kwargs and every query, returns a Tavily-shaped result dict."""

    instances: list[dict] = []
    queries: list[str] = []
    result: dict = {"answer": "Summary text.", "results": [{"title": "T", "url": "https://x"}]}

    def __init__(self, **kwargs):
        _RecordingTavilySearch.instances.append(kwargs)

    def invoke(self, payload):
        _RecordingTavilySearch.queries.append(payload["query"])
        return _RecordingTavilySearch.result


class _Budget:
    def __init__(self, allowance: int):
        self.allowance = allowance
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.calls <= self.allowance


@pytest.fixture
def recording_search(monkeypatch):
    _RecordingTavilySearch.instances = []
    _RecordingTavilySearch.queries = []
    _RecordingTavilySearch.result = {
        "answer": "Summary text.", "results": [{"title": "T", "url": "https://x"}],
    }
    monkeypatch.setattr(research_tools, "TavilySearch", _RecordingTavilySearch)

    def no_deep_research(*args, **kwargs):
        raise AssertionError("trial mode must never call the deep-research endpoint")

    monkeypatch.setattr(research_tools, "TavilyClient", no_deep_research)
    return _RecordingTavilySearch


def _trial_tools(budget):
    return {
        t.name: t for t in research_tools.create_research_tools("AAPL", "fake-key", budget)
    }


@pytest.mark.asyncio
async def test_trial_searches_use_basic_depth(recording_search):
    tools = _trial_tools(_Budget(10))
    for name in ("web_search", "get_company_news", "analyze_competitors", "get_industry_trends"):
        await tools[name].ainvoke("q")
    assert recording_search.instances
    assert {kw["search_depth"] for kw in recording_search.instances} == {"basic"}


@pytest.mark.asyncio
async def test_trial_deep_research_is_a_basic_web_search(recording_search):
    budget = _Budget(10)
    out = await _trial_tools(budget)["deep_research"].ainvoke("supply chain")
    assert "Summary text." in out
    assert recording_search.instances[0]["search_depth"] == "basic"
    assert budget.calls == 1


@pytest.mark.asyncio
async def test_trial_search_refused_when_budget_is_spent(recording_search):
    out = await _trial_tools(_Budget(0))["web_search"].ainvoke("q")
    assert out == research_tools.TRIAL_SEARCH_EXHAUSTED
    assert recording_search.instances == []


@pytest.mark.asyncio
async def test_trial_repeat_search_is_cached_and_not_charged(recording_search):
    budget = _Budget(10)
    tools = _trial_tools(budget)
    first = await tools["web_search"].ainvoke("q")
    second = await tools["web_search"].ainvoke("q")
    assert first == second
    assert budget.calls == 1
    assert len(recording_search.queries) == 1


@pytest.mark.asyncio
async def test_trial_cache_is_keyed_by_tool(recording_search):
    # Same query string, different tool → different Tavily call.
    budget = _Budget(10)
    tools = _trial_tools(budget)
    await tools["analyze_competitors"].ainvoke("")
    await tools["get_industry_trends"].ainvoke("")
    assert budget.calls == 2


@pytest.mark.asyncio
async def test_trial_failed_search_is_not_cached(recording_search, monkeypatch):
    def failing(**kwargs):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(research_tools, "TavilySearch", failing)
    budget = _Budget(10)
    tools = _trial_tools(budget)
    assert (await tools["web_search"].ainvoke("q")).startswith("Failed")

    monkeypatch.setattr(research_tools, "TavilySearch", _RecordingTavilySearch)
    assert "Summary text." in await tools["web_search"].ainvoke("q")
    assert budget.calls == 2


def test_trial_tools_reject_sync_invoke(recording_search):
    # A sync path would skip the async meter entirely.
    with pytest.raises(NotImplementedError):
        _trial_tools(_Budget(10))["web_search"].invoke("q")


def test_non_trial_tools_keep_advanced_depth(recording_search):
    tools = {t.name: t for t in research_tools.create_research_tools("AAPL", "fake-key")}
    tools["web_search"].invoke("q")
    assert recording_search.instances[0]["search_depth"] == "advanced"


class _ErroringTavilySearch:
    """langchain_tavily swallows request failures (429, quota, auth) and
    returns {"error": e} instead of raising."""

    calls = 0

    def __init__(self, **kwargs):
        pass

    def invoke(self, payload):
        type(self).calls += 1
        return {"error": Exception("429 Too Many Requests")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, func",
    [
        ("web_search", lambda q, _retry: research_tools._tool_tavily_search("AAPL", q, "k", "basic")),
        ("get_company_news", lambda q, _retry: research_tools._tool_company_news("AAPL", "k", "basic")),
        ("analyze_competitors", lambda q, _retry: research_tools._tool_competitor_analysis("AAPL", "k", "basic")),
        ("get_industry_trends", lambda q, _retry: research_tools._tool_industry_trends("AAPL", "k", "basic")),
    ],
)
async def test_metered_search_does_not_cache_tavily_errors(monkeypatch, name, func):
    monkeypatch.setattr(research_tools, "TavilySearch", _ErroringTavilySearch)
    _ErroringTavilySearch.calls = 0

    async def budget():
        return True

    out = await research_tools._metered_search("AAPL", name, "news", func, budget)

    assert out.startswith("Failed")
    assert "429" in out
    assert len(research_tools._research_cache) == 0
    # An error on the "day" window must not trigger the "week" retry.
    assert _ErroringTavilySearch.calls == 1


class _QuietDayTavilySearch:
    """No results for the past day (langchain_tavily raises on empty), one
    story for the past week."""

    calls: list = []

    def __init__(self, time_range=None, **kwargs):
        self.time_range = time_range

    def invoke(self, payload):
        type(self).calls.append(self.time_range)
        if self.time_range == "day":
            raise ToolException("No search results found")
        return {"results": [{"title": "Weekly story", "url": "https://example.com/w"}]}


def _counting_budget(allowed: int):
    charged = []

    async def budget():
        charged.append(1)
        return len(charged) <= allowed

    return budget, charged


@pytest.fixture
def _quiet_day(monkeypatch):
    monkeypatch.setattr(research_tools, "TavilySearch", _QuietDayTavilySearch)
    _QuietDayTavilySearch.calls = []


def test_company_news_falls_back_to_week_on_a_quiet_day(_quiet_day):
    out = research_tools._tool_company_news("AAPL", "k", "basic")
    assert "Weekly story" in out
    assert _QuietDayTavilySearch.calls == ["day", "week"]


@pytest.mark.asyncio
async def test_trial_company_news_charges_the_week_fallback(_quiet_day):
    budget, charged = _counting_budget(allowed=5)
    news = next(
        t for t in research_tools.create_research_tools("AAPL", "k", budget)
        if t.name == "get_company_news"
    )

    out = await news.ainvoke("latest")

    assert "Weekly story" in out
    assert len(charged) == 2


@pytest.mark.asyncio
async def test_trial_company_news_skips_fallback_once_budget_is_spent(_quiet_day):
    budget, charged = _counting_budget(allowed=1)
    news = next(
        t for t in research_tools.create_research_tools("AAPL", "k", budget)
        if t.name == "get_company_news"
    )

    out = await news.ainvoke("latest")

    assert out == research_tools.TRIAL_SEARCH_EXHAUSTED
    assert _QuietDayTavilySearch.calls == ["day"]
    assert len(research_tools._research_cache) == 0


class _BusyDayTavilySearch:
    def __init__(self, **kwargs):
        pass

    def invoke(self, payload):
        return {"results": [{"title": "Daily story", "url": "https://example.com/d"}]}


@pytest.mark.asyncio
async def test_trial_rewording_a_query_ignoring_tool_hits_the_cache(monkeypatch):
    monkeypatch.setattr(research_tools, "TavilySearch", _BusyDayTavilySearch)
    budget, charged = _counting_budget(allowed=10)
    news = next(
        t for t in research_tools.create_research_tools("AAPL", "k", budget)
        if t.name == "get_company_news"
    )

    await news.ainvoke("latest news")
    await news.ainvoke("recent developments")

    assert len(charged) == 1
