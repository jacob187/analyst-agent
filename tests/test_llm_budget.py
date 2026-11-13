"""Per-user daily LLM budget tests.

Covers the DB primitive (`increment_llm_usage` / `get_llm_usage`) and the
end-to-end behavior through `check_and_charge_budget`. The HTTP-level
budget enforcement is exercised via the filings endpoint, which is the
canonical operator-paid LLM dispatch site.
"""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import api.db as db_module
from api import llm_concurrency
from api.llm_concurrency import (
    LLMBudgetExceeded,
    LLMCallCharger,
    check_and_charge_budget,
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
# check_and_charge_budget
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
async def test_check_and_charge_budget_anonymous_is_noop(fresh_db):
    # Should not raise and should not write any row.
    await check_and_charge_budget(None)
    # No-op means no row was created — get returns 0 for any user.
    assert await db_module.get_llm_usage(USER_A) == 0


@pytest.mark.eval_unit
async def test_check_and_charge_budget_under_limit(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 5)
    for _ in range(5):
        await check_and_charge_budget(USER_A)
    assert await db_module.get_llm_usage(USER_A) == 5


@pytest.mark.eval_unit
async def test_check_and_charge_budget_exceeded(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await check_and_charge_budget(USER_A)
    await check_and_charge_budget(USER_A)
    # The third call crosses the limit and pays for itself (count == 3).
    with pytest.raises(LLMBudgetExceeded) as exc_info:
        await check_and_charge_budget(USER_A)
    assert exc_info.value.limit == 2
    assert await db_module.get_llm_usage(USER_A) == 3


@pytest.mark.eval_unit
async def test_budget_isolated_per_user(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await check_and_charge_budget(USER_A)
    await check_and_charge_budget(USER_A)
    # A is at the limit; B starts fresh.
    await check_and_charge_budget(USER_B)
    await check_and_charge_budget(USER_B)
    with pytest.raises(LLMBudgetExceeded):
        await check_and_charge_budget(USER_B)
    # A's third call also raises.
    with pytest.raises(LLMBudgetExceeded):
        await check_and_charge_budget(USER_A)


# ---------------------------------------------------------------------------
# Per-call charging: ensure_budget_available + LLMCallCharger
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
async def test_ensure_budget_available_under_limit(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await db_module.increment_llm_usage(USER_A)
    await ensure_budget_available(USER_A)
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
    await ensure_budget_available(None)


def _fake_llm():
    return FakeListChatModel(responses=["ok"])


@pytest.mark.eval_unit
async def test_charger_counts_concurrent_calls_on_the_loop(fresh_db):
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A)
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

    charger = LLMCallCharger(USER_A)
    await RunnableLambda(node).ainvoke("q", config={"callbacks": [charger]})
    assert await charger.settle() == 4
    assert await db_module.get_llm_usage(USER_A) == 4


@pytest.mark.eval_unit
async def test_charger_charges_nothing_when_no_llm_runs(fresh_db):
    charger = LLMCallCharger(USER_A)
    await RunnableLambda(lambda x: x).ainvoke("q", config={"callbacks": [charger]})
    assert await charger.settle() == 0
    assert await db_module.get_llm_usage(USER_A) == 0
