"""Shared fixtures for the eval suite.

Key design decisions:
- `llm` is session-scoped so we reuse one LLM instance across all tests in a run.
  It runs on `EVAL_MODEL_ID`, else the cheapest priced registry model with a
  provider key set; with no key, every test that depends on `llm` auto-skips.
- `planner` binds to AAPL because it's a liquid, well-known ticker with reliable
  SEC filings and Yahoo Finance data — ideal for reproducible evals.
- Dataset loaders are session-scoped and return plain Python dicts from JSON files.
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root so provider API keys are available during tests.
# This runs at import time (before any fixtures) so all fixtures see the vars.
load_dotenv(Path(__file__).parent.parent / ".env")

# Clerk verification (added in Phase 2 of Clerk integration) is enabled when
# CLERK_SECRET_KEY is set in .env for the dev server. Tests should run with
# Clerk disabled by default so legacy integration tests (TestClient + plain
# X-User-Id headers) keep passing. Individual tests opt back in via the
# `clerk_env` fixture in test_dependencies.py.
os.environ.pop("CLERK_SECRET_KEY", None)
os.environ.pop("DISABLE_AUTH", None)


# edgar.set_identity is a process-global setting in edgartools and is normally
# configured once at app startup from api/main.py:lifespan. Pytest doesn't run
# that lifespan for unit tests that import SECDataRetrieval directly, so do it
# here at conftest import time. Uses SEC_HEADER if set (real eval runs), falls
# back to the same dev placeholder api/main.py uses.
def _bootstrap_sec_identity() -> None:
    from edgar import set_identity
    header = (os.environ.get("SEC_HEADER") or "").strip() or "Analyst Agent dev@localhost"
    set_identity(header)


_bootstrap_sec_identity()


LIVE_LLM_MARKERS = ("eval_fast", "eval_tools", "eval_quality", "eval_e2e")


def pytest_collection_modifyitems(config, items):
    """Live-LLM evals run only when `-m` names their marker (e.g. `-m eval_fast`).

    The default addopts only exclude `network`, so with a provider key in `.env`
    a plain `pytest` would otherwise call the real LLM from every eval test (and
    a stalled call hangs the run). Without a key they skip; this makes them
    opt-in either way.
    """
    markexpr = config.getoption("markexpr") or ""
    kept, dropped = [], []
    for item in items:
        live = [m for m in LIVE_LLM_MARKERS if item.get_closest_marker(m)]
        (dropped if live and not any(m in markexpr for m in live) else kept).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


@pytest.fixture(autouse=True)
def _scrub_clerk_env(monkeypatch):
    """Ensure Clerk stays disabled per-test even if api.main re-loads .env.

    `load_dotenv` defaults to override=False, but it re-populates vars we
    popped at conftest import time. monkeypatch removes them for the test
    duration and restores them after.
    """
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_llm_semaphore():
    """Rebuild the module-level LLM dispatch semaphore before each test.

    It's an asyncio.Semaphore that binds to the event loop of its first
    acquire. TestClient runs each request on a fresh loop, so a semaphore
    bound by an earlier test raises 'bound to a different event loop' when a
    later test reuses it. Production runs a single loop, so this is a no-op
    there — purely a test-isolation guard.
    """
    from api import llm_concurrency
    llm_concurrency._rebuild_semaphore()
    yield


@pytest.fixture(scope="session")
def _default_db_guard():
    """Fail the session if any test created the default dev DB (`data/chats.db`).

    Tests must point `api.db.DB_PATH` at a temp file. When the file already
    existed (a dev checkout that has run the app) there is nothing to assert.
    """
    from api import db

    path = db.DB_PATH
    existed = path.exists()
    yield
    assert existed or not path.exists(), (
        f"a test created {path}; point api.db.DB_PATH at a tmp_path file instead"
    )


@pytest.fixture(scope="session", autouse=True)
def _close_db_at_session_end(_default_db_guard):
    """Close any aiosqlite connection a test left open.

    aiosqlite runs each connection on a non-daemon worker thread; one still
    open at interpreter exit (e.g. a test that failed before its fixture
    teardown, or a route that opened the default DB) keeps pytest from exiting.
    Closing from a fresh loop is safe — the thread hands results back
    thread-safely.
    """
    yield
    import asyncio

    from api import db

    if db._db is not None:
        asyncio.run(db.close_db())


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    """Clear in-memory rate-limit caches between tests so REST tests that
    hit the same endpoint repeatedly don't trip each other's quotas."""
    from api.rate_limit import clear_rate_limits

    clear_rate_limits()
    yield
    clear_rate_limits()


@pytest.fixture(autouse=True)
def _reset_sec_analysis_caches():
    """Clear the process-wide SEC retriever / latest-filing caches and the
    per-key asyncio locks between tests: a retriever (or patched fake) cached
    by one test must not serve the next, and a lock used on one test's event
    loop must not be reused on another's."""
    from agents.sec_workflow.analysis_cache import clear_caches

    clear_caches()
    yield
    clear_caches()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).parent / "datasets"
DEFAULT_TICKER = "AAPL"

# SEC filing tool names (13)
# 8-K analysis is a single dispatcher tool (analyze_latest_8k) that routes
# internally to earnings or material event analyzers — replaces the older
# overview / earnings_summary / material_event_summary trio.
SEC_TOOL_NAMES = {
    "get_raw_risk_factors",
    "get_risk_factors_summary",
    "get_raw_management_discussion",
    "get_mda_summary",
    "get_raw_balance_sheets",
    "get_balance_sheet_summary",
    "get_complete_10k_text",
    "get_all_summaries",
    "get_business_overview",
    "get_cybersecurity_disclosure",
    "get_legal_proceedings",
    "analyze_latest_8k",
    "get_8k_item",
}

# Stock / technical analysis tool names (7)
STOCK_TOOL_NAMES = {
    "get_stock_price_history",
    "get_technical_analysis",
    "get_stock_info",
    "get_financial_metrics",
    "get_advanced_technical_analysis",
    "get_pattern_detection",
    "get_multi_timeframe_analysis",
}

# Market-wide context tool names (2)
MARKET_TOOL_NAMES = {
    "get_market_overview",
    "get_macro_indicators",
}

# Research tool names (5) — only available when Tavily key is set
RESEARCH_TOOL_NAMES = {
    "web_search",
    "deep_research",
    "get_company_news",
    "analyze_competitors",
    "get_industry_trends",
}

VALID_TOOLS = SEC_TOOL_NAMES | STOCK_TOOL_NAMES | MARKET_TOOL_NAMES | RESEARCH_TOOL_NAMES


# ---------------------------------------------------------------------------
# LLM fixture — skip if no API key
# ---------------------------------------------------------------------------


def _eval_model():
    """The registry model evals run on: `EVAL_MODEL_ID`, else the cheapest
    priced model whose provider has an env key. Skips when there is none."""
    from agents.model_registry import get_all_models, get_model
    from api.dependencies import PROVIDER_ENV_VARS

    model_id = os.environ.get("EVAL_MODEL_ID")
    if model_id:
        model = get_model(model_id)
        if model is None:
            pytest.skip(f"EVAL_MODEL_ID={model_id!r} is not in agents/models.json")
        return model
    runnable = [
        m for m in get_all_models()
        if m.input_price_per_mtok is not None and os.environ.get(PROVIDER_ENV_VARS[m.provider])
    ]
    if not runnable:
        pytest.skip("no provider API key set for a priced registry model — skipping LLM-dependent test")
    return min(runnable, key=lambda m: (m.input_price_per_mtok + m.output_price_per_mtok, m.id))


@pytest.fixture(scope="session")
def llm():
    """Session-scoped LLM built by `create_llm` from a cheap registry model.

    Automatically skips the test when no provider key is set. This lets
    contributors run the `eval_unit` marker tests without any API keys.
    """
    from functools import partial
    from unittest.mock import patch

    import langchain.chat_models
    from agents.llm_factory import create_llm
    from api.dependencies import PROVIDER_ENV_VARS

    model = _eval_model()
    api_key = os.environ.get(PROVIDER_ENV_VARS[model.provider])
    if not api_key:
        pytest.skip(f"{PROVIDER_ENV_VARS[model.provider]} not set for EVAL_MODEL_ID={model.id!r}")
    # create_llm takes no client options, and provider clients default to no
    # timeout and several retries, so one stalled connection would hang an
    # eval run. Add the limits at the init_chat_model boundary it calls.
    bounded_init = partial(langchain.chat_models.init_chat_model, timeout=60, max_retries=1)
    with patch("langchain.chat_models.init_chat_model", bounded_init):
        return create_llm(model.id, api_key)


# ---------------------------------------------------------------------------
# Planner fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def planner(llm):
    """QueryPlanner bound to AAPL (no research tools)."""
    from agents.planner import QueryPlanner

    return QueryPlanner(llm, ticker=DEFAULT_TICKER, has_research_tools=False)


# ---------------------------------------------------------------------------
# Tools fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tools(llm):
    """List of Tool objects for AAPL (SEC + stock + market)."""
    from agents.tools.sec_tools import create_sec_tools
    from agents.tools.stock_tools import create_stock_tools
    from agents.tools.market_tools import create_market_tools

    tool_list = create_sec_tools(DEFAULT_TICKER, llm)
    tool_list.extend(create_stock_tools(DEFAULT_TICKER))
    tool_list.extend(create_market_tools())
    return tool_list


@pytest.fixture(scope="session")
def tools_dict(tools):
    """Dict mapping tool name → callable for AAPL.

    Each callable drives ``tool.ainvoke`` — the path the graph runtime uses,
    and the only one async-only tools (LLM analyses, DB lookups) support — so
    sync tests and fixtures can keep calling ``tools_dict[name]("")``.
    """
    import asyncio

    def sync_call(tool):
        return lambda tool_input="": asyncio.run(tool.ainvoke(tool_input))

    return {tool.name: sync_call(tool) for tool in tools}


# ---------------------------------------------------------------------------
# Agent fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def agent(llm):
    """Full PlanningAgent for AAPL."""
    from agents.graph.analyst_graph import create_planning_agent

    return create_planning_agent(ticker=DEFAULT_TICKER, llm=llm)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def router_cases():
    """Load router_cases.json — 8 classification test cases."""
    with open(DATASETS_DIR / "router_cases.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def planner_cases():
    """Load planner_cases.json — 5 planning test cases."""
    with open(DATASETS_DIR / "planner_cases.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def golden_queries():
    """Load golden_queries.json — 4 end-to-end quality cases."""
    with open(DATASETS_DIR / "golden_queries.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Provider-agnostic operator-key helpers
#
# Tests of free-trial mechanics, default-model fallback and operator-key
# lending shouldn't care which vendor the registry picks. Derive the model
# from agents.model_registry and use these to wire up its provider's key.
# ---------------------------------------------------------------------------


def set_operator_key(monkeypatch, model, value="operator-key"):
    """Configure the operator's env key for `model`'s provider."""
    from api.dependencies import PROVIDER_ENV_VARS

    monkeypatch.setenv(PROVIDER_ENV_VARS[model.provider], value)


def clear_operator_keys(monkeypatch):
    """Remove every provider's operator env key (no trial, no env fallback)."""
    from api.dependencies import PROVIDER_ENV_VARS

    for env_var in PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)


def byok_auth_field(provider, value="byok-key"):
    """WS auth-frame entry carrying a caller-supplied key for `provider`."""
    from api.dependencies import _PROVIDER_KEY_FIELDS

    return {_PROVIDER_KEY_FIELDS[provider]: value}


def byok_header(provider, value="byok-key"):
    """REST header carrying a caller-supplied key for `provider`."""
    from api.dependencies import _PROVIDER_KEY_FIELDS

    name = "-".join(w.capitalize() for w in _PROVIDER_KEY_FIELDS[provider].split("_"))
    return {f"X-{name}": value}


def non_lendable_model():
    """A registry model the operator's key may not pay for."""
    from agents.model_registry import get_all_models

    return next(m for m in get_all_models() if not m.operator_lendable)


# ---------------------------------------------------------------------------
# Registry-derived model selectors
#
# Tests pick models by property, never by id, so changing the default or
# free-tier model, or adding/removing one in agents/models.json, doesn't break
# the suite. `model_where` / `other_model` skip the calling test when nothing
# matches, so call them inside tests, not at import time.
# ---------------------------------------------------------------------------


def models_where(**fields):
    """Registry models whose ModelDef fields equal every given value."""
    from agents.model_registry import get_all_models

    return [m for m in get_all_models() if all(getattr(m, k) == v for k, v in fields.items())]


def model_where(**fields):
    """The first registry model matching `fields`; skips the test if none does."""
    matches = models_where(**fields)
    if not matches:
        pytest.skip(f"no registry model with {fields}")
    return matches[0]


def other_model(than, **fields):
    """A registry model matching `fields` that isn't `than` (a ModelDef or None)."""
    matches = [m for m in models_where(**fields) if than is None or m.id != than.id]
    if not matches:
        pytest.skip(f"no registry model other than {than and than.id} with {fields}")
    return matches[0]


def one_per_provider(**fields):
    """The first model matching `fields` for each provider that has one (for parametrize)."""
    firsts = {}
    for model in models_where(**fields):
        firsts.setdefault(model.provider, model)
    return list(firsts.values())


def flagged_default_model():
    """The model flagged `default` in models.json, ignoring DEFAULT_MODEL_ID."""
    return models_where(default=True)[0]


def flagged_free_tier_model():
    """The model flagged `free_tier` in models.json (ignoring ANON_FREE_MODEL_ID), or None."""
    return next(iter(models_where(free_tier=True)), None)
