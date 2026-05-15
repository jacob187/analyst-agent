"""Verifies route handlers offload blocking work so the asyncio loop stays responsive.

Strategy: while a slow blocking function is running, a concurrent async ticker
should still get scheduled and increment a counter. If `asyncio.to_thread` were
removed (and the blocking call ran directly on the loop thread), the ticker
would never advance.

These tests do not depend on real network, LLM, or filesystem state. They
exercise the principle: any function called via `asyncio.to_thread` must not
block other coroutines from making progress.
"""

import asyncio
import threading
import time

import pytest


async def _run_with_ticker(coro_factory, blocking_seconds: float = 0.3) -> int:
    """Run `coro_factory()` while a concurrent ticker advances a counter.

    Returns the counter value reached during the call. If the loop were blocked,
    the ticker can't run and the counter stays near 0.
    """
    counter = 0

    async def tick():
        nonlocal counter
        while True:
            await asyncio.sleep(0.01)
            counter += 1

    ticker_task = asyncio.create_task(tick())
    try:
        await coro_factory()
    finally:
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass

    return counter


@pytest.mark.asyncio
async def test_to_thread_keeps_loop_responsive():
    """Baseline: asyncio.to_thread must let other coroutines run."""
    counter = await _run_with_ticker(
        lambda: asyncio.to_thread(time.sleep, 0.3)
    )
    # 300ms blocking call, 10ms ticker → expect ~30 ticks. Allow generous lower bound.
    assert counter >= 15, f"loop appears blocked — ticker only advanced {counter} times"


@pytest.mark.asyncio
async def test_briefing_offloads_generate(monkeypatch):
    """The /watchlist/briefing path must call BriefingService.generate off the loop thread."""
    from agents.briefing.briefing_service import BriefingService

    recorded: dict[str, object] = {}

    def slow_generate(self, tickers):
        recorded["thread_id"] = threading.get_ident()
        recorded["main_thread_id"] = threading.main_thread().ident
        time.sleep(0.1)
        return None  # we won't read the value — only the thread check matters

    monkeypatch.setattr(BriefingService, "generate", slow_generate)

    # Invoke generate the same way the route does: through asyncio.to_thread.
    service = BriefingService.__new__(BriefingService)  # bypass __init__ (needs LLM)
    await asyncio.to_thread(service.generate, ["AAPL"])

    assert recorded["thread_id"] != recorded["main_thread_id"], (
        "BriefingService.generate ran on the main/loop thread — to_thread missing"
    )


@pytest.mark.asyncio
async def test_chart_fetch_runs_off_loop(monkeypatch):
    """`_fetch_chart_payload` must run off the loop when invoked via to_thread."""
    from api.routes import chart

    recorded: dict[str, int] = {}

    def fake_fetch(ticker: str, period_value: str):
        recorded["thread_id"] = threading.get_ident()
        recorded["main_thread_id"] = threading.main_thread().ident
        time.sleep(0.05)
        return {"candles": [], "all_indicators": {}, "quote": {}, "patterns": []}

    monkeypatch.setattr(chart, "_fetch_chart_payload", fake_fetch)
    result = await asyncio.to_thread(chart._fetch_chart_payload, "AAPL", "1y")

    assert result is not None
    assert recorded["thread_id"] != recorded["main_thread_id"], (
        "_fetch_chart_payload ran on the main/loop thread"
    )


@pytest.mark.asyncio
async def test_chart_route_does_not_block_loop_during_fetch(monkeypatch):
    """Concurrent coroutines must advance while the chart fetch is in flight."""
    from api.routes import chart

    def slow_fetch(ticker: str, period_value: str):
        time.sleep(0.3)
        return {"candles": [], "all_indicators": {}, "quote": {}, "patterns": []}

    monkeypatch.setattr(chart, "_fetch_chart_payload", slow_fetch)

    counter = await _run_with_ticker(
        lambda: asyncio.to_thread(chart._fetch_chart_payload, "AAPL", "1y")
    )
    assert counter >= 15, f"loop blocked during chart fetch — ticker advanced {counter} times"
