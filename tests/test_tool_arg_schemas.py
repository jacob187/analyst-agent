"""Tool-call argument tolerance across every tool factory.

Bug: a stronger model called ``get_company_news`` with structured arguments
(``{"ticker": "NVDA", "days": 7}``) and LangChain raised
``ToolException: Too many arguments to single-input tool``. Every tool was a
single-input ``Tool`` taking one string. Tools are now ``StructuredTool``s with
explicit args schemas that ignore unknown arguments, while the graph's
``tool.ainvoke("")`` / single-string call shape keeps working.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import api.db as db_module
from agents.graph.analyst_graph import create_worker_node
from agents.planner import AnalysisStep
from agents.tools import (
    briefing_tools,
    market_tools,
    research_tools,
    sec_tools,
    stock_tools,
)
from tests.sec_fakes import (
    CountingChatModel,
    FakeEdgar,
    FakeEightK,
    FakeFiling,
    eightk,
    tenk,
    tenq,
)

USER_ID = "user_2abcdefghijklmnopqrstuvwxyz"

# Names and descriptions as they were before the Tool -> StructuredTool change.
# Prompts and tool_capabilities.yaml depend on them.
_EXPECTED_DESCRIPTIONS = {
    'get_raw_risk_factors': (
        'Get the complete raw text of Risk Factors section from 10-K filing.'
    ),
    'get_risk_factors_summary': (
        'Get structured analysis of Risk Factors with sentiment and key risks.'
    ),
    'get_raw_management_discussion': (
        'Get raw text of Management Discussion & Analysis (MD&A).'
    ),
    'get_mda_summary': (
        'Get structured MD&A summary with sentiment, outlook, and key points.'
    ),
    'get_raw_balance_sheets': (
        'Get availability of raw balance sheet JSON data (10-K and 10-Q).'
    ),
    'get_balance_sheet_summary': (
        'Get balance sheet summary with key metrics and red flags.'
    ),
    'get_complete_10k_text': (
        'Get list of available 10-K sections retrieved as raw text.'
    ),
    'get_business_overview': (
        'Get the Company Business section (10-K Item 1): products, services, '
        'segments, and market overview.'
    ),
    'get_cybersecurity_disclosure': (
        'Get the Cybersecurity risk management and governance disclosure (10-K '
        'Item 1C, SEC-mandated since 2023).'
    ),
    'get_legal_proceedings': (
        'Get Legal Proceedings section (10-K Item 3): significant pending '
        'litigation, regulatory actions, and legal risks.'
    ),
    'get_all_summaries': (
        'Get a comprehensive overview across risks, MD&A, and financials.'
    ),
    'analyze_latest_8k': (
        'Get structured analysis of the latest 8-K filing. Automatically routes '
        'to earnings analysis (Item 2.02 with a press release) or material event '
        'analysis (leadership changes, agreements, cybersecurity incidents, Reg '
        "FD) based on the filing's contents. Returns sentiment, key points, "
        'impact assessment, and filing metadata as JSON.'
    ),
    'get_8k_item': (
        "Get raw text of a specific 8-K item by number (e.g. '2.02' for earnings, "
        "'1.01' for material agreements, '5.02' for leadership changes). Pass the "
        'item number as `item`.'
    ),
    'get_stock_price_history': (
        'Get recent stock price history (last 10 trading days with open, close, '
        'high, low prices).'
    ),
    'get_technical_analysis': (
        'Get technical indicators including RSI, MACD, Bollinger Bands, moving '
        'averages (5/10/20/50/200-day), volatility, and trend signals.'
    ),
    'get_stock_info': (
        'Get current stock info including current price, trailing P/E ratio, '
        'forward P/E ratio, price-to-book ratio, market cap, 52-week high/low, '
        'volume, average volume, dividend yield, and beta.'
    ),
    'get_financial_metrics': (
        'Get fundamental financial metrics including year-over-year revenue '
        'growth, net income growth, and debt-to-assets ratio from income '
        'statement and balance sheet.'
    ),
    'get_advanced_technical_analysis': (
        'Get advanced technical indicators: ADX (trend strength), ATR '
        '(volatility/position sizing), Stochastic (momentum), Volume Profile '
        '(institutional levels), and Fibonacci retracement levels.'
    ),
    'get_pattern_detection': (
        'Detect chart patterns: Head & Shoulders, Double Top/Bottom, Golden/Death '
        'Cross, RSI divergences. Returns pattern type, confidence, direction, and '
        'price targets.'
    ),
    'get_multi_timeframe_analysis': (
        'Analyze stock across daily, weekly, and hourly timeframes. Detects '
        'conflicts between timeframes and provides weighted trading '
        'recommendation.'
    ),
    'get_market_overview': (
        'Get current levels and daily change for S&P 500, Nasdaq, Dow Jones, and '
        'VIX. Use for broad market context, benchmark comparison, or answering '
        'macro market questions.'
    ),
    'get_macro_indicators': (
        'Get macro economic indicators: 10-Year and 5-Year Treasury yields, VIX '
        'volatility index, and US Dollar Index with 1-month changes. Use for '
        'interest rate context and macro environment analysis.'
    ),
    'web_search': (
        'Search the web for current information about the company. Use this for '
        'general queries, recent events, or fact-checking. Input should be a '
        'search query string.'
    ),
    'deep_research': (
        'Perform comprehensive deep research on a specific topic about the '
        'company. Use this for in-depth analysis, complex questions, or topics '
        'requiring multiple sources. Input should be the research topic.'
    ),
    'get_company_news': (
        'Get the latest news and developments for the company. Returns recent '
        'headlines, news summaries, and source links.'
    ),
    'analyze_competitors': (
        "Research and analyze the company's competitors and market positioning. "
        'Returns competitive landscape analysis and market share information.'
    ),
    'get_industry_trends': (
        'Research industry trends, outlook, and forecasts relevant to the '
        'company. Returns analysis of market trends and industry developments.'
    ),
    'get_briefing_history': (
        'Get recent daily briefing analyses for the company. Returns outlook '
        'trends, technical signals, and news summaries from past morning '
        'briefings. Input: number of days to look back (default 7).'
    ),
    'get_latest_briefing': (
        'Get the most recent full morning briefing summary. Returns market '
        'regime, positioning, all ticker analyses, and alerts.'
    ),
}


FAMILIES = ["sec", "stock", "market", "research", "research_trial", "briefing"]


class _AllowAll:
    async def __call__(self) -> bool:
        return True


def _build(family: str) -> list:
    if family == "sec":
        return sec_tools.create_sec_tools("AAPL", CountingChatModel())
    if family == "stock":
        return stock_tools.create_stock_tools("AAPL")
    if family == "market":
        return market_tools.create_market_tools()
    if family == "research":
        return research_tools.create_research_tools("AAPL", "fake-key")
    if family == "research_trial":
        return research_tools.create_research_tools("AAPL", "fake-key", _AllowAll())
    if family == "briefing":
        return briefing_tools.create_briefing_tools("AAPL", user_id=USER_ID)
    raise AssertionError(family)


_CASES = [(family, tool.name) for family in FAMILIES for tool in _build(family)]
_CASE_IDS = [f"{family}-{name}" for family, name in _CASES]

# Every shape a model or the graph might send: the graph's empty string, a bare
# string, an empty dict, and named / extra arguments in several spellings.
_ALL_ARGS = {
    "ticker": "NVDA",
    "days": 7,
    "max_results": 5,
    "query": "margins",
    "topic": "margins",
    "item": "1.01",
}
_SHAPES = {
    "empty-string": "",
    "single-string": "margins",
    "empty-dict": {},
    "ticker-and-days": {"ticker": "NVDA", "days": 7},
    "query-and-max-results": {"query": "margins", "max_results": 5},
    "everything": _ALL_ARGS,
}


@pytest.fixture(autouse=True)
def _clear_research_cache():
    research_tools._research_cache.clear()
    yield
    research_tools._research_cache.clear()


def _daily_bars(n: int = 400) -> pd.DataFrame:
    """Deterministic OHLCV bars: enough history for MA200 and pattern scans."""
    index = pd.bdate_range(end="2026-11-13", periods=n, tz="America/New_York")
    close = 100 + 10 * np.sin(np.arange(n) / 9) + 0.05 * np.arange(n)
    return pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0,
         "Close": close, "Volume": np.full(n, 1_000_000)},
        index=index,
    )


def _statement(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=["2025-12-31", "2024-12-31"]).T


class _FakeYfTicker:
    """yfinance ``Ticker`` at the external-API boundary."""

    info = {"longName": "Test Co", "currentPrice": 150.0, "previousClose": 148.0,
            "marketCap": 2.5e12, "trailingPE": 28.0, "sector": "Technology"}
    income_stmt = _statement({"Total Revenue": (400e9, 380e9), "Net Income": (100e9, 95e9)})
    balance_sheet = _statement({"Total Assets": (350e9, 340e9), "Total Debt": (100e9, 110e9)})
    calendar: dict = {}

    def __init__(self, symbol: str):
        self.symbol = symbol

    def history(self, period: str = "1y", interval: str = "1d", **_kwargs) -> pd.DataFrame:
        return _daily_bars()


class _FakeTavilyClient:
    """Tavily's research client: completes on the first poll."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def research(self, input: str):
        return {"request_id": "req-1"}

    def get_research(self, request_id: str):
        return {"status": "completed", "content": "Research findings.", "sources": []}


@pytest.fixture
async def boundaries(tmp_path, monkeypatch):
    """Fakes at every external boundary the tool bodies reach, so the real
    implementations run behind the argument layer under test: an in-memory
    EDGAR plus a fake chat model (tests/sec_fakes.py; `_build("sec")`),
    yfinance's ``Ticker``, Tavily's search and research clients, and a temp
    SQLite DB (analysis cache, briefings)."""
    from agents.technical_workflow import indicator_window

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()

    edgar = FakeEdgar()
    edgar.publish(tenk("0000320193-25-000101", "2025-11-01", "10-K risk: supply chain."))
    edgar.publish(tenq("0000320193-26-000111", "2026-02-01", "10-Q risk: new tariffs."))
    edgar.publish(eightk("0000320193-26-000121", "2026-01-30"))
    edgar.install(monkeypatch)

    monkeypatch.setattr("yfinance.Ticker", _FakeYfTicker)
    indicator_window._retriever_cache.clear()
    monkeypatch.setattr(research_tools, "TavilySearch", _FixtureTavilySearch)
    monkeypatch.setattr(research_tools, "TavilyClient", _FakeTavilyClient)
    yield
    indicator_window._retriever_cache.clear()
    await db_module.close_db()


def _assert_ran(name: str, out: str) -> None:
    """The real implementation ran with usable arguments: every tool body
    turns an exception (e.g. a mis-shaped argument) into "Failed to ..."."""
    assert isinstance(out, str), name
    assert not out.startswith("Failed"), (name, out)


# ── AC2: every tool accepts every call shape ─────────────────────────────────


@pytest.mark.parametrize("shape", list(_SHAPES), ids=list(_SHAPES))
@pytest.mark.parametrize(("family", "name"), _CASES, ids=_CASE_IDS)
async def test_every_tool_accepts_every_call_shape(boundaries, family, name, shape):
    tool = {t.name: t for t in _build(family)}[name]
    out = await tool.ainvoke(_SHAPES[shape])
    _assert_ran(name, out)


@pytest.mark.parametrize(("family", "name"), _CASES, ids=_CASE_IDS)
async def test_every_tool_accepts_a_model_tool_call_envelope(boundaries, family, name):
    # What LangGraph's ToolNode sends for a model-emitted call.
    tool = {t.name: t for t in _build(family)}[name]
    message = await tool.ainvoke(
        {"name": name, "args": dict(_ALL_ARGS), "id": "call_1", "type": "tool_call"}
    )
    assert message.status == "success"
    _assert_ran(name, message.content)


@pytest.mark.parametrize(("family", "name"), _CASES, ids=_CASE_IDS)
async def test_tools_are_not_single_input(family, name):
    from langchain_core.tools import StructuredTool

    tool = {t.name: t for t in _build(family)}[name]
    assert isinstance(tool, StructuredTool)


async def test_graph_worker_runs_every_tool_with_its_empty_string_call(boundaries):
    tools = {t.name: t for family in FAMILIES for t in _build(family)}
    worker = create_worker_node(tools)
    for step_id, name in enumerate(tools, start=1):
        step = AnalysisStep(id=step_id, action="x", tool=name, rationale="y")
        result = (await worker({"step": step}))["step_results"][step_id]
        assert not result["error"], (name, result["error"])
        _assert_ran(name, result["raw"])


# ── AC3: names, descriptions, return formats are unchanged ───────────────────


def test_tool_names_and_descriptions_are_unchanged():
    actual = {t.name: t.description for family in FAMILIES for t in _build(family)}
    assert actual == _EXPECTED_DESCRIPTIONS


# ── AC1: get_company_news with structured args (Tavily fixture boundary) ─────

_NEWS_RESPONSE = {
    "answer": "Chipmaker announced a product.",
    "results": [
        {
            "title": "Launch day",
            "url": "https://news.example.com/1",
            "published_date": "Mon, 10 Nov 2026 14:00:00 GMT",
        },
    ],
}

_EXPECTED_NEWS = "\n".join([
    "Recent News for AAPL:",
    "=" * 50,
    "",
    "Summary: Chipmaker announced a product.\n",
    "1. Launch day",
    "   Date: Mon, 10 Nov 2026 14:00:00 GMT",
    "   URL: https://news.example.com/1",
    "",
])


class _FixtureTavilySearch:
    queries: list[str] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, payload):
        type(self).queries.append(payload["query"])
        return _NEWS_RESPONSE


@pytest.fixture
def tavily(monkeypatch):
    _FixtureTavilySearch.queries = []
    monkeypatch.setattr(research_tools, "TavilySearch", _FixtureTavilySearch)
    return _FixtureTavilySearch


@pytest.mark.parametrize("trial", [False, True])
@pytest.mark.parametrize(
    "args",
    [
        {"ticker": "NVDA", "days": 7},
        {"query": "nvidia earnings", "max_results": 5},
        {"ticker": "NVDA", "query": "x", "days": 7, "max_results": 5, "topic": "news"},
        {},
        "",
        "nvidia earnings",
    ],
)
async def test_get_company_news_accepts_structured_arguments(tavily, trial, args):
    tools = {
        t.name: t
        for t in research_tools.create_research_tools(
            "AAPL", "fake-key", _AllowAll() if trial else None
        )
    }
    out = await tools["get_company_news"].ainvoke(args)
    assert out == _EXPECTED_NEWS
    assert len(tavily.queries) == 1


@pytest.mark.parametrize("trial", [False, True])
async def test_web_search_reads_query_from_a_named_argument(tavily, trial):
    tools = {
        t.name: t
        for t in research_tools.create_research_tools(
            "AAPL", "fake-key", _AllowAll() if trial else None
        )
    }
    await tools["web_search"].ainvoke({"query": "margins", "max_results": 3})
    await tools["web_search"].ainvoke("buybacks")
    assert tavily.queries == ["AAPL company margins", "AAPL company buybacks"]


async def test_trial_deep_research_reads_the_topic_argument(tavily):
    tools = {
        t.name: t
        for t in research_tools.create_research_tools("AAPL", "fake-key", _AllowAll())
    }
    await tools["deep_research"].ainvoke({"topic": "supply chain", "depth": "high"})
    assert tavily.queries == ["AAPL company supply chain"]


class _Budget:
    def __init__(self, allowance: int):
        self.allowance = allowance
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.calls <= self.allowance


async def test_trial_metering_still_applies_to_structured_news_calls(tavily):
    budget = _Budget(1)
    tools = {
        t.name: t for t in research_tools.create_research_tools("AAPL", "fake-key", budget)
    }
    first = await tools["get_company_news"].ainvoke({"ticker": "NVDA", "days": 7})
    again = await tools["get_company_news"].ainvoke({"query": "reworded", "days": 3})
    web = await tools["web_search"].ainvoke({"query": "margins"})

    assert first == _EXPECTED_NEWS
    assert again == _EXPECTED_NEWS  # rewording is a cache hit, not a second charge
    assert budget.calls == 2  # news (1) + web_search refused (2)
    assert web == research_tools.TRIAL_SEARCH_EXHAUSTED


# ── Briefing: days arrives as int, numeric string, junk, or not at all ───────


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield
    await db_module.close_db()


@pytest.mark.parametrize(
    ("args", "window"),
    [
        ({"days": 14}, 14),
        ({"days": "14"}, 14),
        ({"days": "soon"}, 7),
        ({"ticker": "AAPL"}, 7),
        ({}, 7),
        ("", 7),
        ("30", 30),
    ],
)
async def test_briefing_history_days_argument(fresh_db, args, window):
    await db_module.save_briefing(
        raw_json="{}",
        market_regime="Bull",
        market_positioning="Long",
        alerts_json="[]",
        thinking=None,
        tickers=[{
            "ticker": "AAPL", "price": 180.5, "change_pct": 1.2, "outlook": "bullish",
            "technical_signal": "MACD cross", "news_summary": "Beat",
        }],
        user_id=USER_ID,
    )
    tools = {t.name: t for t in briefing_tools.create_briefing_tools("AAPL", user_id=USER_ID)}
    out = await tools["get_briefing_history"].ainvoke(args)
    assert f"last {window} days" in out
    assert "BULLISH" in out


async def test_latest_briefing_accepts_named_arguments(fresh_db):
    tools = {t.name: t for t in briefing_tools.create_briefing_tools("AAPL", user_id=USER_ID)}
    out = await tools["get_latest_briefing"].ainvoke({"ticker": "AAPL", "limit": 3})
    assert out == "No briefings have been generated yet."


# ── 8-K item: named, numeric and empty forms (EDGAR retriever boundary) ──────


@pytest.fixture
def edgar(monkeypatch):
    """An in-memory EDGAR whose latest 8-K holds items 1.01, 2.02 and 5.02."""
    items = {item: f"text for item {item}" for item in ("1.01", "2.02", "5.02")}
    filing = FakeFiling(
        "8-K", "0000320193-26-000121", date(2026, 1, 30), FakeEightK(items, True, "2026-01-30"),
    )
    FakeEdgar(filings={"8-K": [filing]}).install(monkeypatch)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"item": "1.01"}, "1.01"),
        ({"item": 5.02}, "5.02"),
        ({"item": "5.02", "ticker": "AAPL", "filing": "latest"}, "5.02"),
        ({}, "2.02"),
        ("1.01", "1.01"),
        ({"query": "5.02"}, "5.02"),
    ],
)
async def test_get_8k_item_argument_forms(edgar, args, expected):
    tools = {t.name: t for t in sec_tools.create_sec_tools("AAPL", object())}
    assert await tools["get_8k_item"].ainvoke(args) == f"text for item {expected}"
