"""Process-wide LLM dispatch concurrency cap + per-user daily budget.

Primitives:

- `llm_slot()` — async context manager backed by a module-level
  `asyncio.Semaphore`. Bounds the number of concurrent in-flight LLM
  dispatches across the whole process. Applies on every deploy.

- `check_and_charge_budget(user_id)` — per-user daily counter backed by
  the `user_llm_usage` SQLite table. Increments today's count and raises
  `LLMBudgetExceeded` once the user crosses `LLM_DAILY_BUDGET`. Only
  meaningful when the operator pays for LLM calls — routes must guard
  with `ApiKeys.is_operator_paid(provider)` so BYOK requests aren't
  charged against the operator's budget.

- Per-call charging: `ensure_budget_available(user_id)` refuses a request up
  front once today's usage is at/over the budget; after that each LLM call is
  charged as it happens — `LLMCallCharger` (a callback on the run config) for
  agent runs, `charge_llm_call` for single direct analyses. The budget unit
  is one LLM call, not one request.

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
LLM_DAILY_BUDGET = int(os.getenv("LLM_DAILY_BUDGET", "100"))


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


async def check_and_charge_budget(user_id: str | None) -> None:
    """Increment user_id's daily LLM count; raise `LLMBudgetExceeded` if over.

    No-op for anonymous traffic (`user_id is None`). Callers must guard
    with `ApiKeys.is_operator_paid(provider)` — BYOK users spend their
    own provider quota and are not charged against the operator's daily
    budget.

    Charge-then-check: the request that crosses the line still pays for
    itself. Matches typical SaaS quota semantics and avoids a TOCTOU race
    under concurrent WS messages.
    """
    if user_id is None:
        return
    count = await increment_llm_usage(user_id)
    if count > LLM_DAILY_BUDGET:
        raise LLMBudgetExceeded(LLM_DAILY_BUDGET)


async def ensure_budget_available(user_id: str | None) -> None:
    """Raise `LLMBudgetExceeded` if user_id has already spent today's budget.

    The pre-flight for per-call charging: a request is refused only when
    usage is already at or over `LLM_DAILY_BUDGET`; once admitted it is
    charged per LLM call (`charge_llm_call` / `LLMCallCharger`) and may
    finish past the line. Concurrent requests can each pass this check, so
    the overshoot is bounded by the calls of the requests in flight.
    No-op for anonymous traffic.
    """
    if user_id is None:
        return
    if await get_llm_usage(user_id) >= LLM_DAILY_BUDGET:
        raise LLMBudgetExceeded(LLM_DAILY_BUDGET)


async def charge_llm_call(user_id: str) -> None:
    """Charge one LLM call to user_id's daily usage."""
    await increment_llm_usage(user_id)


class LLMCallCharger(BaseCallbackHandler):
    """LangChain callback that charges one budget unit per LLM call started.

    Attach it through the run config so every chat-model / LLM start inside
    the run is counted — graph nodes, sync tools in worker threads, and
    `asyncio.to_thread` calls alike (they inherit the run's callbacks via
    contextvars)::

        charger = LLMCallCharger(user_id)
        try:
            async for event in agent.stream(..., config={"callbacks": [charger]}):
                ...
        finally:
            await charger.settle()

    Sync on purpose: LangChain runs an async handler fired from a worker
    thread on a throwaway event loop, where the shared aiosqlite connection
    doesn't belong. Instead each charge is scheduled onto the loop the
    charger was created on, from whichever thread the call started on.
    LLM calls on code paths that drop contextvars (a bare
    ThreadPoolExecutor) don't see the run's callbacks and go uncharged.

    Must be constructed on the running event loop.
    """

    run_inline = True  # fire on the calling thread; scheduling is ours

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.calls = 0
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._pending: list[asyncio.Future | concurrent.futures.Future] = []

    def _charge(self) -> None:
        try:
            try:
                on_loop = asyncio.get_running_loop() is self._loop
            except RuntimeError:
                on_loop = False
            coro = charge_llm_call(self.user_id)
            future: asyncio.Future | concurrent.futures.Future
            if on_loop:
                future = self._loop.create_task(coro)
            else:
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            with self._lock:
                self.calls += 1
                self._pending.append(future)
        except Exception:
            # Never break the LLM call over accounting.
            logger.exception("Failed to schedule LLM budget charge")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._charge()

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        self._charge()

    async def settle(self) -> int:
        """Await every scheduled charge; return the number of calls counted."""
        with self._lock:
            pending = [
                asyncio.wrap_future(f) if isinstance(f, concurrent.futures.Future) else f
                for f in self._pending
            ]
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error("LLM budget charge failed for a call: %r", result)
        return self.calls
