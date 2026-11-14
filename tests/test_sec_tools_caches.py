"""Tests for bounded module-level caches in the tool modules.

Verifies that `_shared_retrievers` / `_latest_filings` and the per-key lock
stores (sec_workflow.analysis_cache — shared by the SEC tools and the filings
endpoint), `_research_cache` (research_tools), and `_retriever_cache`
(indicator_window — shared between chart and stock tools) are size-capped via
`cachetools` so unique-key rotation cannot grow process memory without bound.
SEC analyses themselves live in the `filing_analyses` table, not in-process.
"""

from cachetools import LRUCache, TTLCache

from agents.sec_workflow import analysis_cache
from agents.tools import research_tools
from agents.technical_workflow import indicator_window


class TestSecAnalysisCachesBounded:
    def test_shared_retrievers_bounded_at_128(self):
        for i in range(1000):
            analysis_cache._shared_retrievers[f"T{i}"] = object()
        assert len(analysis_cache._shared_retrievers) <= 128

    def test_latest_filings_bounded_at_1024(self):
        for i in range(5000):
            analysis_cache._latest_filings[(f"T{i}", "10-K")] = (None, 0.0, None)
        assert len(analysis_cache._latest_filings) <= 1024

    def test_lock_stores_bounded(self):
        for i in range(5000):
            analysis_cache._lock_for(analysis_cache._analysis_locks, (f"T{i}",))
        assert len(analysis_cache._analysis_locks) <= 4096

    def test_caches_are_cachetools_types(self):
        assert isinstance(analysis_cache._shared_retrievers, LRUCache)
        assert isinstance(analysis_cache._latest_filings, LRUCache)
        assert isinstance(analysis_cache._latest_locks, TTLCache)
        assert isinstance(analysis_cache._analysis_locks, TTLCache)


class TestResearchCacheBounded:
    def setup_method(self):
        research_tools._research_cache.clear()

    def test_research_cache_bounded_at_256(self):
        for i in range(1000):
            research_tools._research_cache[f"q{i}"] = {"content": "x", "sources": 1}
        assert len(research_tools._research_cache) <= 256

    def test_clear_research_cache_preserves_type(self):
        research_tools._research_cache["q"] = {"content": "x", "sources": 1}
        research_tools.clear_research_cache()
        assert len(research_tools._research_cache) == 0
        # The bug we're guarding against: rebinding `_research_cache = {}` would
        # silently drop the bounded type. Confirm clear() keeps it a TTLCache.
        assert isinstance(research_tools._research_cache, TTLCache)
        assert research_tools._research_cache.ttl == 900


class TestStockRetrieverCacheBounded:
    """Retriever cache lives in indicator_window now — shared between chart
    route and LLM tools so there's a single yfinance.Ticker lifecycle per symbol."""

    def setup_method(self):
        indicator_window._retriever_cache.clear()

    def test_retriever_cache_bounded_at_128(self):
        for i in range(1000):
            indicator_window._retriever_cache[f"T{i}"] = object()
        assert len(indicator_window._retriever_cache) <= 128

    def test_cache_is_cachetools_type(self):
        assert isinstance(indicator_window._retriever_cache, LRUCache)
