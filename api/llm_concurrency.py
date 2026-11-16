"""Process-wide LLM dispatch concurrency cap + per-user daily budget.

Primitives:

- `llm_slot()` — async context manager backed by a module-level
  `asyncio.Semaphore`. Bounds the number of concurrent in-flight LLM
  dispatches across the whole process. Applies on every deploy.

- Per-user daily budget, backed by the `user_llm_usage` SQLite table. Only
  meaningful when the operator pays for LLM calls — routes guard with
  `ApiKeys.is_operator_paid(provider)` so BYOK requests aren't charged.
  `ensure_budget_available(user_id)` refuses a request up front once today's
  usage is at/over the budget and returns the usage it saw; after that each
  LLM call is charged as it happens — `LLMCallCharger` (a callback on the run
  config, which also stops the run once the budget is spent) for agent runs,
  `charge_llm_call` for single direct analyses. The budget unit is one LLM
  call, not one request.

The cap and budget are env-tunable. Tests override the module constants via
`monkeypatch.setattr` and call `_rebuild_semaphore()` to pick up a new
size.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from api.db import get_llm_usage, increment_llm_usage

logger = logging.getLogger(__name__)


LLM_DISPATCH_CONCURRENCY = int(os.getenv("LLM_DISPATCH_CONCURRENCY", "8"))
LLM_DISPATCH_WAIT_SECONDS = float(os.getenv("LLM_DISPATCH_WAIT_SECONDS", "30"))
# LLM calls per signed-in user per UTC day on the operator's key. The unit was
# one request until charging moved to one per LLM call; a request averaged
# ~3.5 calls in the benchmark (router + answer + summary, more with tools or
# planning), so 350 keeps roughly the former 100-request allowance.
LLM_DAILY_BUDGET = int(os.getenv("LLM_DAILY_BUDGET", "350"))


_semaphore = asyncio.Semaphore(LLM_DISPATCH_CONCURRENCY)


class LLMConcurrencyExceeded(RuntimeError):
    """Raised when the LLM dispatch pool stays saturated past the wait timeout."""


class LLMBudgetExceeded(RuntimeError):
    """Raised when a user crosses their daily LLM dispatch budget."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"Daily LLM budget of {limit} exceeded")


def _rebuild_semaphore() -> None:
    """Reset the module semaphore to the current `LLM_DISPATCH_CONCURRENCY`.

    Tests mutate the constant via `monkeypatch.setattr` and then call this
    to install a fresh semaphore. Not for production use.
    """
    global _semaphore
    _semaphore = asyncio.Semaphore(LLM_DISPATCH_CONCURRENCY)


@asynccontextmanager
async def llm_slot():
    """Acquire a process-wide LLM dispatch slot.

    Waits up to `LLM_DISPATCH_WAIT_SECONDS` for the semaphore. Raises
    `LLMConcurrencyExceeded` if the pool stays saturated — routes map this
    to HTTP 503 / a WS error event so callers fail fast rather than queue
    behind a stuck dispatcher.
    """
    try:
        await asyncio.wait_for(_semaphore.acquire(), timeout=LLM_DISPATCH_WAIT_SECONDS)
    except asyncio.TimeoutError as e:
        logger.warning(
            "LLM dispatch pool saturated: waited %ss, cap=%s",
            LLM_DISPATCH_WAIT_SECONDS,
            LLM_DISPATCH_CONCURRENCY,
        )
        raise LLMConcurrencyExceeded(
            f"LLM dispatch pool saturated (cap={LLM_DISPATCH_CONCURRENCY})"
        ) from e
    try:
        yield
    finally:
        _semaphore.release()


async def ensure_budget_available(user_id: str | None) -> int:
    """Raise `LLMBudgetExceeded` if user_id has already spent today's budget.

    The pre-flight for per-call charging: a request is refused when usage is
    already at or over `LLM_DAILY_BUDGET`. Returns the usage it read (0 for
    anonymous traffic, which is never charged) so an `LLMCallCharger` can
    start from it.
    """
    if user_id is None:
        return 0
    used = await get_llm_usage(user_id)
    if used >= LLM_DAILY_BUDGET:
        raise LLMBudgetExceeded(LLM_DAILY_BUDGET)
    return used


async def charge_llm_call(user_id: str) -> None:
    """Charge one LLM call to user_id's daily usage."""
    await increment_llm_usage(user_id)


class LLMCallCharger(BaseCallbackHandler):
    """LangChain callback that charges one budget unit per LLM call started,
    and stops the run once the user's daily budget is spent.

    Attach it through the run config so every chat-model / LLM start inside
    the run is counted — graph nodes, sync tools in worker threads, and
    `asyncio.to_thread` calls alike (they inherit the run's callbacks via
    contextvars)::

        used = await ensure_budget_available(user_id)
        charger = LLMCallCharger(user_id, used=used)
        try:
            async for event in agent.stream(..., config={"callbacks": [charger]}):
                ...
        finally:
            await charger.settle()

    Enforcement: the charger tracks today's usage, starting from `used` (what
    the pre-flight read; routes always pass it, 0 means unknown) plus its own
    calls, and raised to the count each DB
    increment returns, which includes other requests' charges. A call that
    starts once that reaches `LLM_DAILY_BUDGET` raises `LLMBudgetExceeded`
    from the LLM call (`raise_error`), so the run stops. The check never
    waits on the DB, so it can't block or deadlock whichever thread the call
    starts on; the cost is that concurrent requests learn of each other's
    charges only when their own land, so the overshoot is bounded by the
    calls in flight (two requests admitted at budget - 1 make one call each).

    Sync on purpose: LangChain runs an async handler fired from a worker
    thread on a throwaway event loop, where the shared aiosqlite connection
    doesn't belong. Instead each charge is scheduled onto the loop the
    charger was created on, from whichever thread the call started on, and
    a failed charge is logged when it completes — also for a call that
    starts after `settle()` (a worker thread still running past a route's
    timeout), which nothing awaits. LLM calls on code paths that drop
    contextvars (a bare ThreadPoolExecutor) don't see the run's callbacks
    and go uncharged.

    Must be constructed on the running event loop.
    """

    run_inline = True  # fire on the calling thread; scheduling is ours
    raise_error = True  # LLMBudgetExceeded must reach the LLM call

    def __init__(self, user_id: str, *, used: int = 0) -> None:
        self.user_id = user_id
        self.calls = 0
        self._used = used
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._pending: list[concurrent.futures.Future] = []

    def _charge(self) -> None:
        with self._lock:
            if self._used >= LLM_DAILY_BUDGET:
                raise LLMBudgetExceeded(LLM_DAILY_BUDGET)
            self._used += 1
            self.calls += 1
        coro = increment_llm_usage(self.user_id)
        try:
            # Non-blocking from any thread, the loop's own included.
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception:
            # Never break the LLM call over accounting (e.g. the loop closed).
            coro.close()
            logger.exception("Failed to schedule LLM budget charge")
            return
        with self._lock:
            self._pending.append(future)
        # Outside the lock: a future that is already done runs this inline.
        future.add_done_callback(self._charged)

    def _charged(self, future: concurrent.futures.Future) -> None:
        if future.cancelled():
            logger.error("LLM budget charge cancelled for user call")
            return
        exc = future.exception()
        if exc is not None:
            logger.error("LLM budget charge failed for a call: %r", exc)
            return
        with self._lock:
            self._used = max(self._used, future.result())

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._charge()

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        self._charge()

    async def settle(self) -> int:
        """Await every charge scheduled so far; return the calls counted.

        Failures are logged by the charge itself, not here.
        """
        with self._lock:
            pending = [asyncio.wrap_future(f) for f in self._pending]
        await asyncio.gather(*pending, return_exceptions=True)
        return self.calls
