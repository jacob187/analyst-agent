"""Bounds tests for company route in-memory state: profile-locks and stream
semaphores.

Pre-fix: `_profile_locks` was a plain `dict` and `_stream_semaphores` was a
`defaultdict(lambda: asyncio.Semaphore(...))`. Both grew without bound — every
novel ticker or IP added an entry that was never reclaimed, giving any user
or scanner a memory amplification vector.

Post-fix: both are `cachetools.TTLCache(maxsize=4096, ttl=600)` accessed via
`_get_profile_lock` / `_get_stream_semaphore` helpers. The TTL drops
unreferenced entries; in-flight requests holding a Python reference on their
stack keep the object alive even after eviction.
"""

import asyncio

import pytest
from cachetools import TTLCache

from api.routes import company


@pytest.fixture(autouse=True)
def _clear_company_caches():
    company._profile_locks.clear()
    company._stream_semaphores.clear()
    yield
    company._profile_locks.clear()
    company._stream_semaphores.clear()


class TestProfileLocks:
    def test_is_ttl_cache_with_expected_bounds(self):
        assert isinstance(company._profile_locks, TTLCache)
        assert company._profile_locks.maxsize == 4096
        assert company._profile_locks.ttl == 600

    def test_get_returns_same_object_within_ttl(self):
        a = company._get_profile_lock("AAPL")
        b = company._get_profile_lock("AAPL")
        assert a is b, "Same ticker must yield the same Lock within TTL"

    def test_distinct_tickers_get_distinct_locks(self):
        a = company._get_profile_lock("AAPL")
        b = company._get_profile_lock("MSFT")
        assert a is not b

    def test_bounded_under_ticker_rotation(self):
        for i in range(10_000):
            company._get_profile_lock(f"T{i}")
        assert len(company._profile_locks) <= 4096

    def test_returns_asyncio_lock_instance(self):
        lock = company._get_profile_lock("AAPL")
        assert isinstance(lock, asyncio.Lock)


class TestStreamSemaphores:
    def test_is_ttl_cache_with_expected_bounds(self):
        assert isinstance(company._stream_semaphores, TTLCache)
        assert company._stream_semaphores.maxsize == 4096
        assert company._stream_semaphores.ttl == 600

    def test_get_returns_same_object_for_same_ip(self):
        a = company._get_stream_semaphore("1.2.3.4")
        b = company._get_stream_semaphore("1.2.3.4")
        assert a is b

    def test_distinct_ips_get_distinct_semaphores(self):
        a = company._get_stream_semaphore("1.2.3.4")
        b = company._get_stream_semaphore("5.6.7.8")
        assert a is not b

    def test_bounded_under_ip_rotation(self):
        for i in range(10_000):
            company._get_stream_semaphore(f"10.0.{i // 256}.{i % 256}")
        assert len(company._stream_semaphores) <= 4096

    def test_semaphore_has_expected_capacity(self):
        sem = company._get_stream_semaphore("1.2.3.4")
        # Default Semaphore.acquire — we don't expose _value, so check by counting.
        assert isinstance(sem, asyncio.Semaphore)
        # Can acquire `_MAX_CONCURRENT_STREAMS` times without blocking.
        loop = asyncio.new_event_loop()
        try:
            for _ in range(company._MAX_CONCURRENT_STREAMS):
                loop.run_until_complete(asyncio.wait_for(sem.acquire(), 0.1))
            # Next acquire should block (hit capacity).
            with pytest.raises(asyncio.TimeoutError):
                loop.run_until_complete(asyncio.wait_for(sem.acquire(), 0.05))
        finally:
            loop.close()
