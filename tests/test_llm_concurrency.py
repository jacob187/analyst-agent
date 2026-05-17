"""LLM dispatch concurrency tests.

`llm_slot()` wraps every actual LLM call via the analyst_graph
`_run_with_timeout` helper plus the two `_run_llm_analysis` sites in
`api/routes/company.py`. The semaphore caps process-wide in-flight
dispatches; saturation past the wait timeout raises
`LLMConcurrencyExceeded`.
"""

import asyncio

import pytest

from api import llm_concurrency
from api.llm_concurrency import LLMConcurrencyExceeded, llm_slot


@pytest.fixture
def cap_3(monkeypatch):
    """Shrink the semaphore to size 3 for the duration of the test."""
    monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_CONCURRENCY", 3)
    llm_concurrency._rebuild_semaphore()
    yield
    monkeypatch.undo()
    llm_concurrency._rebuild_semaphore()


@pytest.mark.asyncio
async def test_llm_slot_acquires_and_releases():
    async with llm_slot():
        pass
    # If release didn't happen we'd block here forever.
    async with llm_slot():
        pass


@pytest.mark.asyncio
async def test_llm_slot_caps_in_flight_at_n(cap_3):
    """5 concurrent acquirers must observe a peak in-flight count of 3."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()
    started = asyncio.Event()

    async def worker():
        nonlocal in_flight, peak
        async with llm_slot():
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            if not started.is_set():
                started.set()
            # Give other workers time to also try to acquire.
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(5)))
    assert peak == 3, f"expected peak 3, got {peak}"
    assert in_flight == 0


@pytest.mark.asyncio
async def test_llm_slot_raises_on_saturation_timeout(monkeypatch):
    """When the pool is full and the wait timeout expires, the next
    acquire raises LLMConcurrencyExceeded instead of blocking forever."""
    monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_CONCURRENCY", 1)
    monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_WAIT_SECONDS", 0.05)
    llm_concurrency._rebuild_semaphore()

    holder_release = asyncio.Event()

    async def holder():
        async with llm_slot():
            await holder_release.wait()

    holder_task = asyncio.create_task(holder())
    # Yield so the holder grabs the slot before we test contention.
    await asyncio.sleep(0)

    with pytest.raises(LLMConcurrencyExceeded):
        async with llm_slot():
            pass  # never reached

    holder_release.set()
    await holder_task
    monkeypatch.undo()
    llm_concurrency._rebuild_semaphore()


@pytest.mark.asyncio
async def test_llm_slot_releases_on_exception(cap_3):
    """An exception inside the CM body must still release the slot —
    otherwise one bad call permanently shrinks the pool."""
    for _ in range(5):
        with pytest.raises(RuntimeError, match="boom"):
            async with llm_slot():
                raise RuntimeError("boom")
    # If a slot leaked, we'd block here after 3 successful acquires.
    async with llm_slot():
        async with llm_slot():
            async with llm_slot():
                pass
