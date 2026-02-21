"""Shared fixtures for the eval suite.

Key design decisions:
- `llm` is session-scoped so we reuse one LLM instance across all tests in a run.
  If GOOGLE_API_KEY is missing, every test that depends on `llm` auto-skips.
- `planner` binds to AAPL because it's a liquid, well-known ticker with reliable
  SEC filings and Yahoo Finance data — ideal for reproducible evals.
- Dataset loaders are session-scoped and return plain Python dicts from JSON files.
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root so GOOGLE_API_KEY is available during tests.
# This runs at import time (before any fixtures) so all fixtures see the vars.
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).parent / "datasets"
DEFAULT_TICKER = "AAPL"

# All SEC tool names (12) — no research tools
SEC_TOOL_NAMES = {
    "get_raw_risk_factors",
    "get_risk_factors_summary",
    "get_raw_management_discussion",
    "get_mda_summary",
    "get_raw_balance_sheets",
    "get_balance_sheet_summary",
    "get_complete_10k_text",
    "get_all_summaries",
    "get_stock_price_history",
    "get_technical_analysis",
    "get_stock_info",
    "get_financial_metrics",
}

# Research tool names (5) — only available when Tavily key is set
RESEARCH_TOOL_NAMES = {
    "web_search",
    "deep_research",
    "get_company_news",
    "analyze_competitors",
    "get_industry_trends",
}

VALID_TOOLS = SEC_TOOL_NAMES | RESEARCH_TOOL_NAMES


# ---------------------------------------------------------------------------
# LLM fixture — skip if no API key
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llm():
    """Session-scoped Gemini LLM with temperature=0 for determinism.

    Automatically skips the test if GOOGLE_API_KEY is not set. This lets
    contributors run the `eval_unit` marker tests without any API keys.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set — skipping LLM-dependent test")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


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
    """List of Tool objects for AAPL."""
    from agents.tools.sec_tools import create_sec_tools

    tool_list, _ = create_sec_tools(DEFAULT_TICKER, llm)
    return tool_list


@pytest.fixture(scope="session")
def tools_dict(tools):
    """Dict mapping tool name → callable for AAPL."""
    return {tool.name: tool.func for tool in tools}


# ---------------------------------------------------------------------------
# Agent fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def agent(llm):
    """Full PlanningAgent for AAPL."""
    from agents.graph.sec_graph import create_planning_agent

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
