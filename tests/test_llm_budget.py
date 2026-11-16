"""Per-user daily LLM budget tests.

Covers the DB primitive (`increment_llm_usage` / `get_llm_usage`), the
request pre-flight (`ensure_budget_available`) and per-call charging with
enforcement (`LLMCallCharger`). Route-level behaviour is exercised in the
chat, watchlist and filings endpoint tests.
"""

import asyncio
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import api.db as db_module
from api import llm_concurrency
from api.llm_concurrency import (
    LLMBudgetExceeded,
    LLMCallCharger,
    ensure_budget_available,
)


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    """Same isolation pattern as tests/test_db.py — fresh SQLite per test."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_db", None)
    await db_module.init_db()
    yield
    await db_module.close_db()


# ---------------------------------------------------------------------------
# DB primitive
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
async def test_increment_llm_usage_counts_up(fresh_db):
    assert await db_module.increment_llm_usage(USER_A) == 1
    assert await db_module.increment_llm_usage(USER_A) == 2
    assert await db_module.increment_llm_usage(USER_A) == 3


@pytest.mark.eval_unit
async def test_increment_llm_usage_isolated_per_user(fresh_db):
    await db_module.increment_llm_usage(USER_A)
    await db_module.increment_llm_usage(USER_A)
    assert await db_module.increment_llm_usage(USER_B) == 1
    # A's count untouched
    assert await db_module.get_llm_usage(USER_A) == 2


@pytest.mark.eval_unit
async def test_concurrent_increments_each_see_their_own_count(fresh_db):
    # Parallel planner workers charge the trial search meter at once; each
    # must get a distinct count or two can be refused at the cap boundary.
    counts = await asyncio.gather(*(db_module.increment_llm_usage(USER_A) for _ in range(20)))
    assert sorted(counts) == list(range(1, 21))


@pytest.mark.eval_unit
async def test_get_llm_usage_zero_for_fresh_user(fresh_db):
    assert await db_module.get_llm_usage(USER_A) == 0


@pytest.mark.eval_unit
async def test_get_llm_usage_rolls_over_next_day(fresh_db, monkeypatch):
    monkeypatch.setattr(db_module, "_today_utc", lambda: "2026-05-17")
    await db_module.increment_llm_usage(USER_A)
    await db_module.increment_llm_usage(USER_A)
    assert await db_module.get_llm_usage(USER_A) == 2

    # Advance the clock — yesterday's row is preserved but today is fresh.
    monkeypatch.setattr(db_module, "_today_utc", lambda: "2026-05-18")
    assert await db_module.get_llm_usage(USER_A) == 0


# ---------------------------------------------------------------------------
# Default budget
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
def test_default_budget_is_sized_for_per_call_charging():
    # A fresh interpreter: reloading the module in-process would replace
    # LLMBudgetExceeded under the routes that catch it.
    env = {k: v for k, v in os.environ.items() if k != "LLM_DAILY_BUDGET"}
    result = subprocess.run(
        [sys.executable, "-c", "import api.llm_concurrency as m; print(m.LLM_DAILY_BUDGET)"],
        cwd=Path(__file__).resolve().parent.parent, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.split()[-1]) == 350


# ---------------------------------------------------------------------------
# Pre-flight: ensure_budget_available
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
async def test_ensure_budget_available_under_limit_returns_usage(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await db_module.increment_llm_usage(USER_A)
    assert await ensure_budget_available(USER_A) == 1
    # The check itself charges nothing.
    assert await db_module.get_llm_usage(USER_A) == 1


@pytest.mark.eval_unit
async def test_ensure_budget_available_refuses_at_limit(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await db_module.increment_llm_usage(USER_A)
    await db_module.increment_llm_usage(USER_A)
    with pytest.raises(LLMBudgetExceeded):
        await ensure_budget_available(USER_A)
    assert await db_module.get_llm_usage(USER_A) == 2


@pytest.mark.eval_unit
async def test_ensure_budget_available_anonymous_is_noop(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 0)
    assert await ensure_budget_available(None) == 0


# ---------------------------------------------------------------------------
# Per-call charging: LLMCallCharger
# ---------------------------------------------------------------------------


def _fake_llm():
    return FakeListChatModel(responses=["ok"])


@pytest.mark.eval_unit
async def test_charger_counts_concurrent_calls_on_the_loop(fresh_db):
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=0)
    config = {"callbacks": [charger]}
    await asyncio.gather(*(llm.ainvoke("hi", config=config) for _ in range(5)))
    assert await charger.settle() == 5
    assert await db_module.get_llm_usage(USER_A) == 5


@pytest.mark.eval_unit
async def test_charger_counts_calls_made_from_worker_threads(fresh_db):
    # Graph nodes run sync LLM calls via asyncio.to_thread and inherit the
    # run's callbacks through contextvars; those charges must land on the
    # owning loop's DB connection.
    llm = _fake_llm()

    async def node(_x):
        await asyncio.gather(*(asyncio.to_thread(llm.invoke, "hi") for _ in range(3)))
        await llm.ainvoke("hi")
        return "done"

    charger = LLMCallCharger(USER_A, used=0)
    await RunnableLambda(node).ainvoke("q", config={"callbacks": [charger]})
    assert await charger.settle() == 4
    assert await db_module.get_llm_usage(USER_A) == 4


@pytest.mark.eval_unit
async def test_charger_charges_nothing_when_no_llm_runs(fresh_db):
    charger = LLMCallCharger(USER_A, used=0)
    await RunnableLambda(lambda x: x).ainvoke("q", config={"callbacks": [charger]})
    assert await charger.settle() == 0
    assert await db_module.get_llm_usage(USER_A) == 0


@pytest.mark.eval_unit
async def test_charger_stops_an_on_loop_run_once_the_budget_is_spent(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await db_module.increment_llm_usage(USER_A)
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=await ensure_budget_available(USER_A))
    config = {"callbacks": [charger]}

    await llm.ainvoke("hi", config=config)  # usage 1 -> 2: the last allowed call
    with pytest.raises(LLMBudgetExceeded):
        await llm.ainvoke("hi", config=config)

    assert await charger.settle() == 1
    assert await db_module.get_llm_usage(USER_A) == 2


@pytest.mark.eval_unit
async def test_charger_stops_a_worker_thread_call_once_the_budget_is_spent(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 1)
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=0)
    config = {"callbacks": [charger]}

    await asyncio.to_thread(llm.invoke, "hi", config)
    with pytest.raises(LLMBudgetExceeded):
        await asyncio.to_thread(llm.invoke, "hi", config)

    assert await charger.settle() == 1
    assert await db_module.get_llm_usage(USER_A) == 1


@pytest.mark.eval_unit
async def test_charger_learns_of_other_runs_charges(fresh_db, monkeypatch):
    # Admitted at 0, but another request spends 2 units meanwhile: this run's
    # own charge returns the true count, and its next call is refused.
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 3)
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=0)
    config = {"callbacks": [charger]}
    await db_module.increment_llm_usage(USER_A)
    await db_module.increment_llm_usage(USER_A)

    await llm.ainvoke("hi", config=config)
    await charger.settle()  # the charge has landed: usage is 3
    with pytest.raises(LLMBudgetExceeded):
        await llm.ainvoke("hi", config=config)
    assert await db_module.get_llm_usage(USER_A) == 3


@pytest.mark.eval_unit
async def test_concurrent_runs_overshoot_by_at_most_their_in_flight_calls(fresh_db, monkeypatch):
    # Two requests admitted at budget - 1 each start one call before either
    # charge lands: both run (the bound is the calls in flight), then both
    # runs are stopped.
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 3)
    for _ in range(2):
        await db_module.increment_llm_usage(USER_A)
    llm = _fake_llm()
    chargers = [LLMCallCharger(USER_A, used=await ensure_budget_available(USER_A)) for _ in range(2)]

    await asyncio.gather(*(llm.ainvoke("hi", config={"callbacks": [c]}) for c in chargers))
    for c in chargers:
        assert await c.settle() == 1
        with pytest.raises(LLMBudgetExceeded):
            await llm.ainvoke("hi", config={"callbacks": [c]})
    assert await db_module.get_llm_usage(USER_A) == 4


@pytest.mark.eval_unit
def test_sync_call_on_the_loop_thread_does_not_deadlock(tmp_path, monkeypatch):
    """A sync `invoke` made on the loop thread itself (blocking the loop) fires
    the callback there; the charger must not wait on the loop it is blocking.
    Run in a daemon thread so a regression fails the join, not the suite."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "loop.db")
    monkeypatch.setattr(db_module, "_db", None)
    result: dict[str, int] = {}

    async def main():
        await db_module.init_db()
        try:
            charger = LLMCallCharger(USER_A, used=0)
            _fake_llm().invoke("hi", config={"callbacks": [charger]})
            result["calls"] = await charger.settle()
            result["usage"] = await db_module.get_llm_usage(USER_A)
        finally:
            await db_module.close_db()

    worker = threading.Thread(target=asyncio.run, args=(main(),), daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "charger deadlocked on the loop thread"
    assert result == {"calls": 1, "usage": 1}


@pytest.mark.eval_unit
async def test_charge_failing_after_settle_is_logged(fresh_db, monkeypatch, caplog):
    """A call that starts after the route settled (a briefing thread still
    running past its 504) is charged by a task nobody awaits; if that charge
    fails it must still be logged."""
    charger = LLMCallCharger(USER_A, used=0)
    assert await charger.settle() == 0

    async def failing_increment(user_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(llm_concurrency, "increment_llm_usage", failing_increment)
    with caplog.at_level(logging.ERROR, logger="api.llm_concurrency"):
        await asyncio.to_thread(_fake_llm().invoke, "hi", {"callbacks": [charger]})
        for _ in range(20):
            if caplog.records:
                break
            await asyncio.sleep(0.01)
    assert any("database is locked" in r.getMessage() for r in caplog.records)
