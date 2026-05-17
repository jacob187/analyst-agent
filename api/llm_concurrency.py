"""Process-wide LLM dispatch concurrency cap + per-user daily budget.

Two independent primitives:

- `llm_slot()` — async context manager backed by a module-level
  `asyncio.Semaphore`. Bounds the number of concurrent in-flight LLM
  dispatches across the whole process. Applies on every deploy.

- `check_and_charge_budget(user_id)` — per-user daily counter backed by
  the `user_llm_usage` SQLite table. Increments today's count and raises
  `LLMBudgetExceeded` once the user crosses `LLM_DAILY_BUDGET`. Only
  meaningful when the operator pays for LLM calls — routes must guard
  with `ApiKeys.is_operator_paid(provider)` so BYOK requests aren't
  charged against the operator's budget.

Both are env-tunable. Tests override the module constants via
`monkeypatch.setattr` and call `_rebuild_semaphore()` to pick up a new
size.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from api.db import increment_llm_usage

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
