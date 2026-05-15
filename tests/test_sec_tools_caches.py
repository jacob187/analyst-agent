"""Tests for bounded module-level caches in the tool modules.

Verifies that `_shared_retrievers`, `_shared_processors`, `_processed_cache`
(sec_tools), `_research_cache` (research_tools), and `_shared_stock_retrievers`
(stock_tools) are size-capped via `cachetools` so unique-key rotation cannot
grow process memory without bound.
"""

from cachetools import LRUCache, TTLCache

from agents.tools import research_tools, sec_tools, stock_tools


class TestSecToolCachesBounded:
    def setup_method(self):
        sec_tools._shared_retrievers.clear()
        sec_tools._shared_processors.clear()
        sec_tools._processed_cache.clear()

    def test_shared_retrievers_bounded_at_128(self):
        for i in range(1000):
            sec_tools._shared_retrievers[f"T{i}"] = object()
        assert len(sec_tools._shared_retrievers) <= 128

    def test_shared_processors_bounded_at_128(self):
        for i in range(1000):
            sec_tools._shared_processors[f"PROC{i}"] = object()
        assert len(sec_tools._shared_processors) <= 128

    def test_processed_cache_outer_bounded_at_512(self):
        for i in range(1000):
            sec_tools._processed_cache[f"T{i}"] = {"risk_summary": {"v": i}}
        assert len(sec_tools._processed_cache) <= 512

    def test_processed_cache_setdefault_survives_outer_eviction(self):
        # Simulate the race the setdefault fix protects against:
        # outer TTL evicts the ticker between the read-check and the write.
        sec_tools._processed_cache["AAPL"] = {"risk_summary": {"v": 1}}
        del sec_tools._processed_cache["AAPL"]  # simulate TTL eviction
        # The production write site uses setdefault, so this must not raise.
        sec_tools._processed_cache.setdefault("AAPL", {})["mda_summary"] = {"v": 2}
        assert sec_tools._processed_cache["AAPL"] == {"mda_summary": {"v": 2}}

    def test_processed_cache_ttl_forces_eviction(self):
        # cachetools.TTLCache.expire(time) evicts entries whose expiry time
        # is at or before the passed clock value. We push the clock past TTL
        # to verify eviction without sleeping.
        sec_tools._processed_cache["AAPL"] = {"risk_summary": {"v": 1}}
        assert "AAPL" in sec_tools._processed_cache
        future = sec_tools._processed_cache.timer() + sec_tools._processed_cache.ttl + 1
        sec_tools._processed_cache.expire(future)
        assert "AAPL" not in sec_tools._processed_cache

    def test_caches_are_cachetools_types(self):
        assert isinstance(sec_tools._shared_retrievers, LRUCache)
        assert isinstance(sec_tools._shared_processors, LRUCache)
        assert isinstance(sec_tools._processed_cache, TTLCache)
        assert sec_tools._processed_cache.ttl == 3600


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
    def setup_method(self):
        stock_tools._shared_stock_retrievers.clear()

    def test_shared_stock_retrievers_bounded_at_128(self):
        for i in range(1000):
            stock_tools._shared_stock_retrievers[f"T{i}"] = object()
        assert len(stock_tools._shared_stock_retrievers) <= 128

    def test_cache_is_cachetools_type(self):
        assert isinstance(stock_tools._shared_stock_retrievers, LRUCache)
