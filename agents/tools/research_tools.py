"""Tavily research tools for deep web research on companies and financial topics."""

from typing import Dict, Any, Optional
from langchain_core.tools import Tool

# Cache for research results to minimize API calls
_research_cache: Dict[str, Dict[str, Any]] = {}


def _get_cache_key(ticker: str, query: str) -> str:
    """Generate cache key for research results."""
    return f"{ticker}_{hash(query)}"


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
        from langchain_tavily import TavilySearch

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


def _tool_tavily_research(
    ticker: str,
    topic: str,
    tavily_api_key: str,
    model: str = "pro",
) -> str:
    """
    Perform deep research on a company topic using Tavily's Research API.

    This provides comprehensive, multi-source research with citations.

    Args:
        ticker: Company ticker symbol
        topic: Research topic
        tavily_api_key: Tavily API key
        model: "mini" (quick), "pro" (comprehensive), or "auto"
    """
    cache_key = _get_cache_key(ticker, topic)

    # Check cache first
    if cache_key in _research_cache:
        cached = _research_cache[cache_key]
        return f"[Cached Research]\n\n{cached.get('content', 'No content')}"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_api_key)

        # Create research request with company context
        research_query = f"Comprehensive analysis of {ticker}: {topic}"

        # Start research
        response = client.research(
            query=research_query,
            max_results=10,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )

        # Process the research response
        if isinstance(response, dict):
            answer = response.get("answer", "")
            results = response.get("results", [])

            output_lines = [f"Deep Research Report: {ticker} - {topic}", "=" * 60, ""]

            if answer:
                output_lines.append("Executive Summary:")
                output_lines.append("-" * 40)
                output_lines.append(answer)
                output_lines.append("")

            if results:
                output_lines.append("Key Sources & Findings:")
                output_lines.append("-" * 40)
                for i, r in enumerate(results[:7], 1):
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    content = r.get("content", "")[:300]
                    output_lines.append(f"\n{i}. {title}")
                    output_lines.append(f"   Source: {url}")
                    output_lines.append(f"   Key Points: {content}...")

            result_text = "\n".join(output_lines)

            # Cache the result
            _research_cache[cache_key] = {
                "content": result_text,
                "sources": len(results),
            }

            return result_text
        else:
            return str(response)

    except ImportError:
        # Fallback to langchain-tavily if tavily-python not available
        return _tool_tavily_search(ticker, topic, tavily_api_key, "advanced")
    except Exception as e:
        return f"Failed to perform deep research: {e}"


def _tool_company_news(ticker: str, tavily_api_key: str) -> str:
    """
    Get recent news and developments for a company.

    Args:
        ticker: Company ticker symbol
        tavily_api_key: Tavily API key
    """
    try:
        from langchain_tavily import TavilySearch

        search = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=7,
            search_depth="advanced",
            include_answer=True,
            topic="news",
        )

        query = f"{ticker} stock latest news developments 2025"
        result = search.invoke({"query": query})

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
        from langchain_tavily import TavilySearch

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
        from langchain_tavily import TavilySearch

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
    global _research_cache
    _research_cache = {}
