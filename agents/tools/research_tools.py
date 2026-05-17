"""Tavily research tools for deep web research on companies and financial topics."""

import asyncio
import hashlib
from concurrent.futures import (
    ThreadPoolExecutor, TimeoutError as FuturesTimeout,
)
from typing import Any, Optional

from cachetools import TTLCache
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from tavily import TavilyClient

# Bounded research cache: 15-min TTL matches briefing-cache convention.
_research_cache: TTLCache = TTLCache(maxsize=256, ttl=900)

# Tavily deep-research polling configuration.
_RESEARCH_POLL_INTERVAL_SECONDS = 2.0
_RESEARCH_MAX_POLLS = 30

# Per-call wall-clock cap on TavilySearch.invoke(). langchain-tavily's
# underlying requests.post() has no timeout argument, so we enforce one
# externally via a Future. A hung Tavily request returns None to the
# caller (treated as a soft failure) instead of blocking the tool for the
# full ~75s OS-level socket timeout.
_TAVILY_PER_CALL_TIMEOUT_SECONDS = 10


def _invoke_with_timeout(search: Any, query: str, timeout_s: float) -> Optional[Any]:
    """Run search.invoke({"query": query}) with an external wall-clock cap.

    Returns the result, or None on timeout. The hung thread is detached
    (not killed) so the agent step can finish on schedule.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(search.invoke, {"query": query})
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _get_cache_key(ticker: str, query: str) -> str:
    """Generate a stable cache key for research results.

    Uses sha1 over the encoded query so keys are reproducible across process
    restarts (Python's builtin hash() is salted per process via PYTHONHASHSEED).
    Truncated to 16 hex chars — ample collision resistance for a per-process cache.
    """
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    return f"{ticker}_{digest}"


def _tool_tavily_search(
    ticker: str, query: str, tavily_api_key: str, search_depth: str = "advanced"
) -> str:
    """
    Perform a Tavily web search for company-related information.

    Args:
        ticker: Company ticker symbol
        query: Search query
        tavily_api_key: Tavily API key
        search_depth: "basic" for quick search, "advanced" for comprehensive
    """
    try:
        search = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=5,
            search_depth=search_depth,
            include_answer=True,
        )

        # Enhance query with company context
        enhanced_query = f"{ticker} company {query}"
        result = search.invoke({"query": enhanced_query})

        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            answer = result.get("answer", "")
            results = result.get("results", [])

            output_lines = []
            if answer:
                output_lines.append(f"Summary: {answer}\n")

            output_lines.append("Sources:")
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                content = r.get("content", "")[:200]
                output_lines.append(f"\n{i}. {title}")
                output_lines.append(f"   URL: {url}")
                output_lines.append(f"   {content}...")

            return "\n".join(output_lines)
        else:
            return str(result)

    except Exception as e:
        return f"Failed to perform web search: {e}"


def _format_research_result(ticker: str, topic: str, result: dict) -> str:
    content = result.get("content", "")
    sources = result.get("sources", [])

    output_lines = [f"Deep Research Report: {ticker} - {topic}", "=" * 60, "", content]

    if sources:
        output_lines.extend(["", "Sources:", "-" * 40])
        for i, source in enumerate(sources[:10], 1):
            output_lines.append(f"{i}. {source.get('title', 'No title')}")
            output_lines.append(f"   {source.get('url', '')}")

    return "\n".join(output_lines)


async def _tool_tavily_research_async(
    ticker: str,
    topic: str,
    tavily_api_key: str,
) -> str:
    """Async variant of `_tool_tavily_research`.

    The Tavily research endpoint returns a request_id immediately and then
    requires the caller to poll until status=="completed" (or "failed"). The
    sync version slept the worker thread for up to 60s; this version releases
    the thread between polls so the asyncio thread pool stays free for other
    tools (charts, briefing, SEC analysis).

    HTTP calls themselves stay sync via `asyncio.to_thread` since the Tavily
    SDK does not expose async methods.
    """
    cache_key = _get_cache_key(ticker, topic)

    if cache_key in _research_cache:
        cached = _research_cache[cache_key]
        return f"[Cached Research]\n\n{cached.get('content', 'No content')}"

    try:
        client = TavilyClient(api_key=tavily_api_key)
        research_query = f"Comprehensive analysis of {ticker}: {topic}"

        response = await asyncio.to_thread(client.research, input=research_query)
        request_id = response.get("request_id")

        if not request_id:
            return "Failed to initiate deep research: no request_id returned"

        for _ in range(_RESEARCH_MAX_POLLS):
            result = await asyncio.to_thread(client.get_research, request_id)
            status = result.get("status", "")

            if status == "completed":
                result_text = _format_research_result(ticker, topic, result)
                sources = result.get("sources", [])
                _research_cache[cache_key] = {"content": result_text, "sources": len(sources)}
                return result_text

            if status == "failed":
                return f"Deep research failed: {result.get('error', 'Unknown error')}"

            await asyncio.sleep(_RESEARCH_POLL_INTERVAL_SECONDS)

        return "Deep research timed out. Please try again."

    except Exception as e:
        return f"Failed to perform deep research: {e}"


def _tool_tavily_research(
    ticker: str,
    topic: str,
    tavily_api_key: str,
) -> str:
    """Sync fallback for legacy callers (`Tool.invoke`). Inside the LangGraph
    worker the async coroutine is always preferred via `Tool.ainvoke`."""
    return asyncio.run(_tool_tavily_research_async(ticker, topic, tavily_api_key))


def _tool_company_news(ticker: str, tavily_api_key: str) -> str:
    """
    Get recent news and developments for a company.

    Args:
        ticker: Company ticker symbol
        tavily_api_key: Tavily API key
    """
    try:
        query = f"{ticker} stock latest news developments"

        # Freshness: day-first, fall back to week so quiet days still return
        # something. Without time_range Tavily ranks purely by relevance and
        # cheerfully returns week-old "top" articles.
        # Each call is wrapped in _invoke_with_timeout so a hung Tavily
        # endpoint can't pin the agent step.
        result: dict | str | None = None
        for time_range in ("day", "week"):
            search = TavilySearch(
                tavily_api_key=tavily_api_key,
                max_results=7,
                search_depth="advanced",
                include_answer=True,
                topic="news",
                time_range=time_range,
            )
            result = _invoke_with_timeout(
                search, query, _TAVILY_PER_CALL_TIMEOUT_SECONDS,
            )
            if isinstance(result, dict) and result.get("results"):
                break
        if result is None:
            return f"News retrieval timed out for {ticker}."

        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            answer = result.get("answer", "")
            results = result.get("results", [])

            output_lines = [f"Recent News for {ticker}:", "=" * 50, ""]

            if answer:
                output_lines.append(f"Summary: {answer}\n")

            for i, r in enumerate(results[:7], 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                published = r.get("published_date", "")
                output_lines.append(f"{i}. {title}")
                if published:
                    output_lines.append(f"   Date: {published}")
                output_lines.append(f"   URL: {url}")
                output_lines.append("")

            return "\n".join(output_lines)
        else:
            return str(result)

    except Exception as e:
        return f"Failed to retrieve news: {e}"


def _tool_competitor_analysis(ticker: str, tavily_api_key: str) -> str:
    """
    Research competitors and market positioning for a company.

    Args:
        ticker: Company ticker symbol
        tavily_api_key: Tavily API key
    """
    try:
        search = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=10,
            search_depth="advanced",
            include_answer=True,
        )

        query = f"{ticker} competitors market share industry analysis comparison"
        result = search.invoke({"query": query})

        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            answer = result.get("answer", "")
            results = result.get("results", [])

            output_lines = [
                f"Competitor Analysis for {ticker}:",
                "=" * 50,
                "",
            ]

            if answer:
                output_lines.append("Market Position Summary:")
                output_lines.append("-" * 30)
                output_lines.append(answer)
                output_lines.append("")

            output_lines.append("Research Sources:")
            output_lines.append("-" * 30)
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                content = r.get("content", "")[:200]
                output_lines.append(f"\n{i}. {title}")
                output_lines.append(f"   {content}...")

            return "\n".join(output_lines)
        else:
            return str(result)

    except Exception as e:
        return f"Failed to analyze competitors: {e}"


def _tool_industry_trends(ticker: str, tavily_api_key: str) -> str:
    """
    Research industry trends and outlook relevant to the company.

    Args:
        ticker: Company ticker symbol
        tavily_api_key: Tavily API key
    """
    try:
        search = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=8,
            search_depth="advanced",
            include_answer=True,
        )

        query = f"{ticker} industry trends outlook forecast market analysis 2025"
        result = search.invoke({"query": query})

        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            answer = result.get("answer", "")
            results = result.get("results", [])

            output_lines = [
                f"Industry Trends & Outlook for {ticker}:",
                "=" * 50,
                "",
            ]

            if answer:
                output_lines.append("Industry Overview:")
                output_lines.append("-" * 30)
                output_lines.append(answer)
                output_lines.append("")

            output_lines.append("Key Sources:")
            output_lines.append("-" * 30)
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                content = r.get("content", "")[:200]
                output_lines.append(f"\n{i}. {title}")
                output_lines.append(f"   {content}...")

            return "\n".join(output_lines)
        else:
            return str(result)

    except Exception as e:
        return f"Failed to analyze industry trends: {e}"


def create_research_tools(ticker: str, tavily_api_key: str) -> list[Tool]:
    """
    Create Tavily research tools bound to a specific ticker.

    Args:
        ticker: Company ticker symbol
        tavily_api_key: Tavily API key

    Returns:
        List of LangChain Tool objects for research
    """
    tools: list[Tool] = [
        Tool.from_function(
            name="web_search",
            description=(
                "Search the web for current information about the company. "
                "Use this for general queries, recent events, or fact-checking. "
                "Input should be a search query string."
            ),
            func=lambda query: _tool_tavily_search(ticker, query, tavily_api_key),
        ),
        Tool.from_function(
            name="deep_research",
            description=(
                "Perform comprehensive deep research on a specific topic about the company. "
                "Use this for in-depth analysis, complex questions, or topics requiring multiple sources. "
                "Input should be the research topic."
            ),
            func=lambda topic: _tool_tavily_research(ticker, topic, tavily_api_key),
            coroutine=lambda topic: _tool_tavily_research_async(ticker, topic, tavily_api_key),
        ),
        Tool.from_function(
            name="get_company_news",
            description=(
                "Get the latest news and developments for the company. "
                "Returns recent headlines, news summaries, and source links."
            ),
            func=lambda query="": _tool_company_news(ticker, tavily_api_key),
        ),
        Tool.from_function(
            name="analyze_competitors",
            description=(
                "Research and analyze the company's competitors and market positioning. "
                "Returns competitive landscape analysis and market share information."
            ),
            func=lambda query="": _tool_competitor_analysis(ticker, tavily_api_key),
        ),
        Tool.from_function(
            name="get_industry_trends",
            description=(
                "Research industry trends, outlook, and forecasts relevant to the company. "
                "Returns analysis of market trends and industry developments."
            ),
            func=lambda query="": _tool_industry_trends(ticker, tavily_api_key),
        ),
    ]

    return tools


def clear_research_cache() -> None:
    """Clear all cached research results."""
    _research_cache.clear()
