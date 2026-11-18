"""Per-user daily LLM budget tests.

Covers the DB primitive (`increment_llm_usage` / `get_llm_usage`), the
request pre-flight (`ensure_budget_available`) and per-call charging with
enforcement (`LLMCallCharger`). Route-level behaviour is exercised in the
chat, watchlist and filings endpoint tests.
"""

import asyncio
import logging
import math
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import api.db as db_module
from agents.model_registry import get_default_model
from api import llm_concurrency
from api.llm_concurrency import (
    OPERATOR_GLOBAL_USAGE_KEY as GLOBAL,
    BudgetUsage,
    LLMBudgetExceeded,
    LLMCallCharger,
    LLMGlobalBudgetExceeded,
    ensure_budget_available,
)
from tests.conftest import models_where, one_per_provider


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
NONE_USED = BudgetUsage(user=0, overall=0)


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
async def test_increment_llm_usage_adds_an_amount_in_one_update(fresh_db):
    # A weighted call is charged in one statement, so concurrent callers each
    # see a count that includes their whole charge.
    assert await db_module.increment_llm_usage(USER_A, 3) == 3
    assert await db_module.increment_llm_usage(USER_A, 2) == 5
    assert await db_module.get_llm_usage(USER_A) == 5


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


@pytest.mark.eval_unit
async def test_increment_llm_usage_many_returns_each_new_count(fresh_db):
    assert await db_module.increment_llm_usage_many({USER_A: 3, GLOBAL: 3}) == {USER_A: 3, GLOBAL: 3}
    assert await db_module.increment_llm_usage_many({USER_B: 2, GLOBAL: 2}) == {USER_B: 2, GLOBAL: 5}


async def _refuse_global_row_writes() -> None:
    """Make every write to the global row fail, as a crash or cancellation
    between a charge's two rows would."""
    db = await db_module.get_db()
    await db.execute(
        "CREATE TRIGGER refuse_global BEFORE INSERT ON user_llm_usage "
        "WHEN NEW.user_id = 'operator:global' BEGIN SELECT RAISE(ABORT, 'global row refused'); END"
    )
    await db.commit()


@pytest.mark.eval_unit
async def test_increment_llm_usage_many_commits_all_rows_or_none(fresh_db):
    assert GLOBAL == "operator:global"  # the trigger below names it
    await _refuse_global_row_writes()
    with pytest.raises(sqlite3.IntegrityError):
        await db_module.increment_llm_usage_many({USER_A: 3, GLOBAL: 3})
    assert await db_module.get_llm_usage(USER_A) == 0


@pytest.mark.eval_unit
async def test_a_charge_that_fails_on_the_global_row_leaves_the_user_row_alone(fresh_db):
    # Charged on both rows or neither: the global cap never counts less than
    # the users' rows add up to.
    await _refuse_global_row_writes()
    charger = LLMCallCharger(USER_A, used=NONE_USED)
    await _fake_llm().ainvoke("hi", config={"callbacks": [charger]})
    await charger.settle()
    assert await db_module.get_llm_usage(USER_A) == 0
    assert await db_module.get_llm_usage(GLOBAL) == 0


# ---------------------------------------------------------------------------
# Default budget
# ---------------------------------------------------------------------------


_SETTINGS = (
    "LLM_DAILY_BUDGET", "LLM_GLOBAL_DAILY_BUDGET", "LLM_USER_GLOBAL_SHARE",
    "LLM_DISPATCH_CONCURRENCY", "LLM_DISPATCH_WAIT_SECONDS",
)


def _settings_in_fresh_interpreter(**overrides: str) -> list[str]:
    """The budget/dispatch settings as a fresh interpreter imports them with
    `overrides` in the environment (reloading the module in-process would
    replace LLMBudgetExceeded under the routes that catch it)."""
    env = {k: v for k, v in os.environ.items() if k not in _SETTINGS}
    env.update(overrides)
    result = subprocess.run(
        [sys.executable, "-c",
         "import api.llm_concurrency as m; "
         "print(m.LLM_DAILY_BUDGET, m.LLM_GLOBAL_DAILY_BUDGET, m.LLM_USER_GLOBAL_SHARE, "
         "m.LLM_DISPATCH_CONCURRENCY, m.LLM_DISPATCH_WAIT_SECONDS)"],
        cwd=Path(__file__).resolve().parent.parent, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()[-5:]


_DEFAULT_SETTINGS = ["350", "1500", "0.02", "8", "30.0"]


@pytest.mark.eval_unit
def test_default_budget_is_sized_for_per_call_charging():
    assert _settings_in_fresh_interpreter() == _DEFAULT_SETTINGS


@pytest.mark.eval_unit
@pytest.mark.parametrize("raw_int, raw_float", [("", ""), ("3.5e2", "nan"), ("abc", "abc"), ("-1", "-1")])
def test_malformed_budget_settings_fall_back_to_defaults(raw_int, raw_float):
    """A blank or malformed value must not crash backend import."""
    overrides = {
        "LLM_DAILY_BUDGET": raw_int,
        "LLM_GLOBAL_DAILY_BUDGET": raw_int,
        "LLM_USER_GLOBAL_SHARE": raw_float,
        "LLM_DISPATCH_CONCURRENCY": raw_int,
        "LLM_DISPATCH_WAIT_SECONDS": raw_float,
    }
    assert _settings_in_fresh_interpreter(**overrides) == _DEFAULT_SETTINGS


@pytest.mark.eval_unit
@pytest.mark.parametrize("raw, expected", [
    ("0", "0.02"), ("1.5", "0.02"), ("-0.1", "0.02"), ("inf", "0.02"),
    ("0.5", "0.5"), ("1", "1.0"),
])
def test_user_global_share_must_be_a_fraction_of_the_pool(raw, expected):
    # 0 would refuse every user; over 1 is no limit at all. Both fall back.
    assert _settings_in_fresh_interpreter(LLM_USER_GLOBAL_SHARE=raw)[2] == expected


@pytest.mark.eval_unit
def test_zero_dispatch_concurrency_falls_back_to_default():
    # A zero-slot pool would refuse every LLM call.
    assert _settings_in_fresh_interpreter(LLM_DISPATCH_CONCURRENCY="0")[3] == "8"


# ---------------------------------------------------------------------------
# Pre-flight: ensure_budget_available
# ---------------------------------------------------------------------------


@pytest.mark.eval_unit
async def test_ensure_budget_available_under_limit_returns_usage(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await db_module.increment_llm_usage(USER_A)
    await db_module.increment_llm_usage(GLOBAL, 4)
    assert await ensure_budget_available(USER_A) == BudgetUsage(user=1, overall=4)
    # The check itself charges nothing.
    assert await db_module.get_llm_usage(USER_A) == 1
    assert await db_module.get_llm_usage(GLOBAL) == 4


@pytest.mark.eval_unit
async def test_ensure_budget_available_refuses_at_limit(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
    await db_module.increment_llm_usage(USER_A)
    await db_module.increment_llm_usage(USER_A)
    with pytest.raises(LLMBudgetExceeded) as refused:
        await ensure_budget_available(USER_A)
    assert not isinstance(refused.value, LLMGlobalBudgetExceeded)
    assert await db_module.get_llm_usage(USER_A) == 2


@pytest.mark.eval_unit
async def test_ensure_budget_available_refuses_once_everyone_has_spent_the_global_budget(
    fresh_db, monkeypatch,
):
    # Many accounts, each under its own budget, still add up to one cap.
    monkeypatch.setattr(llm_concurrency, "LLM_GLOBAL_DAILY_BUDGET", 3)
    await db_module.increment_llm_usage(GLOBAL, 3)
    with pytest.raises(LLMGlobalBudgetExceeded) as refused:
        await ensure_budget_available(USER_A)
    assert "free daily allowance" in str(refused.value)
    assert "your own API key" in str(refused.value)


@pytest.mark.eval_unit
async def test_zero_global_budget_lends_nothing(fresh_db, monkeypatch):
    # 0 means no operator-paid calls at all, as LLM_DAILY_BUDGET=0 does.
    monkeypatch.setattr(llm_concurrency, "LLM_GLOBAL_DAILY_BUDGET", 0)
    with pytest.raises(LLMGlobalBudgetExceeded):
        await ensure_budget_available(USER_A)


@pytest.mark.eval_unit
async def test_ensure_budget_available_anonymous_is_noop(fresh_db, monkeypatch):
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 0)
    monkeypatch.setattr(llm_concurrency, "LLM_GLOBAL_DAILY_BUDGET", 0)
    assert await ensure_budget_available(None) == BudgetUsage(user=0, overall=0)


@pytest.mark.eval_unit
def test_refusal_message_tells_the_user_which_budget_ran_out():
    own = llm_concurrency.refusal_message(LLMBudgetExceeded(5))
    everyones = llm_concurrency.refusal_message(LLMGlobalBudgetExceeded(1500))
    assert own == llm_concurrency.BUDGET_REACHED
    assert everyones == llm_concurrency.GLOBAL_BUDGET_REACHED
    assert "your own API key" in own and "your own API key" in everyones


def _share_of_pool(monkeypatch, *, daily: int, pool: int, share: float) -> None:
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", daily)
    monkeypatch.setattr(llm_concurrency, "LLM_GLOBAL_DAILY_BUDGET", pool)
    monkeypatch.setattr(llm_concurrency, "LLM_USER_GLOBAL_SHARE", share)


@pytest.mark.eval_unit
@pytest.mark.parametrize("daily, pool, share, limit", [
    (350, 1500, 0.02, 30),   # the share binds
    (20, 1500, 0.02, 20),    # the user budget binds
    (350, 10, 0.02, 1),      # a share under one unit still allows a call
    (350, 0, 0.02, 350),     # no pool to take a share of
])
def test_user_daily_budget_is_capped_at_a_share_of_the_pool(monkeypatch, daily, pool, share, limit):
    _share_of_pool(monkeypatch, daily=daily, pool=pool, share=share)
    assert llm_concurrency.user_daily_budget() == limit


@pytest.mark.eval_unit
async def test_ensure_budget_available_refuses_a_user_at_their_share_of_the_pool(fresh_db, monkeypatch):
    # Under LLM_DAILY_BUDGET, but at 5% of a 100-unit pool: refused as the
    # user's own budget (not everyone's), and another user is unaffected.
    _share_of_pool(monkeypatch, daily=350, pool=100, share=0.05)
    await db_module.increment_llm_usage(USER_A, 5)
    await db_module.increment_llm_usage(GLOBAL, 5)
    with pytest.raises(LLMBudgetExceeded) as refused:
        await ensure_budget_available(USER_A)
    assert not isinstance(refused.value, LLMGlobalBudgetExceeded)
    assert refused.value.limit == 5
    assert await ensure_budget_available(USER_B) == BudgetUsage(user=0, overall=5)


# ---------------------------------------------------------------------------
# Per-call charging: LLMCallCharger
# ---------------------------------------------------------------------------


def _fake_llm():
    return FakeListChatModel(responses=["ok"])


@pytest.mark.eval_unit
async def test_charger_counts_concurrent_calls_on_the_loop(fresh_db):
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=NONE_USED)
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

    charger = LLMCallCharger(USER_A, used=NONE_USED)
    await RunnableLambda(node).ainvoke("q", config={"callbacks": [charger]})
    assert await charger.settle() == 4
    assert await db_module.get_llm_usage(USER_A) == 4


@pytest.mark.eval_unit
async def test_charger_charges_nothing_when_no_llm_runs(fresh_db):
    charger = LLMCallCharger(USER_A, used=NONE_USED)
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
    charger = LLMCallCharger(USER_A, used=NONE_USED)
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
    charger = LLMCallCharger(USER_A, used=NONE_USED)
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
async def test_every_users_charges_add_to_the_global_usage(fresh_db):
    llm = _fake_llm()
    for user in (USER_A, USER_B):
        charger = LLMCallCharger(user, used=await ensure_budget_available(user))
        await llm.ainvoke("hi", config={"callbacks": [charger]})
        await llm.ainvoke("hi", config={"callbacks": [charger]})
        assert await charger.settle() == 2
    assert await db_module.get_llm_usage(USER_A) == 2
    assert await db_module.get_llm_usage(USER_B) == 2
    assert await db_module.get_llm_usage(GLOBAL) == 4


@pytest.mark.eval_unit
async def test_charger_stops_a_user_at_their_share_of_the_pool(fresh_db, monkeypatch):
    # Draining a 10-unit pool at a 0.5 share takes two accounts: the first
    # is stopped at 5 units, the second still runs.
    _share_of_pool(monkeypatch, daily=350, pool=10, share=0.5)
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=await ensure_budget_available(USER_A))
    config = {"callbacks": [charger]}
    for _ in range(5):
        await llm.ainvoke("hi", config=config)
    with pytest.raises(LLMBudgetExceeded) as refused:
        await llm.ainvoke("hi", config=config)
    assert not isinstance(refused.value, LLMGlobalBudgetExceeded)
    assert await charger.settle() == 5

    other = LLMCallCharger(USER_B, used=await ensure_budget_available(USER_B))
    await llm.ainvoke("hi", config={"callbacks": [other]})
    assert await other.settle() == 1
    assert await db_module.get_llm_usage(GLOBAL) == 6


@pytest.mark.eval_unit
async def test_charger_stops_a_run_once_the_global_budget_is_spent(fresh_db, monkeypatch):
    # USER_B is far under their own budget; everyone else's usage stops them.
    # A share of 1 lets one user spend the whole (tiny) pool.
    _share_of_pool(monkeypatch, daily=350, pool=3, share=1.0)
    await db_module.increment_llm_usage(GLOBAL, 2)
    llm = _fake_llm()
    charger = LLMCallCharger(USER_B, used=await ensure_budget_available(USER_B))
    config = {"callbacks": [charger]}

    await llm.ainvoke("hi", config=config)  # global 2 -> 3: the last allowed call
    with pytest.raises(LLMGlobalBudgetExceeded):
        await llm.ainvoke("hi", config=config)

    assert await charger.settle() == 1
    assert await db_module.get_llm_usage(USER_B) == 1
    assert await db_module.get_llm_usage(GLOBAL) == 3


@pytest.mark.eval_unit
async def test_charger_learns_of_other_users_global_charges(fresh_db, monkeypatch):
    _share_of_pool(monkeypatch, daily=350, pool=3, share=1.0)
    llm = _fake_llm()
    charger = LLMCallCharger(USER_A, used=await ensure_budget_available(USER_A))
    config = {"callbacks": [charger]}
    await db_module.increment_llm_usage(GLOBAL, 2)  # another user, meanwhile

    await llm.ainvoke("hi", config=config)
    await charger.settle()  # the charge has landed: global usage is 3
    with pytest.raises(LLMGlobalBudgetExceeded):
        await llm.ainvoke("hi", config=config)
    assert await db_module.get_llm_usage(GLOBAL) == 3


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
            charger = LLMCallCharger(USER_A, used=NONE_USED)
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
    charger = LLMCallCharger(USER_A, used=NONE_USED)
    assert await charger.settle() == 0

    async def failing_increment(amounts):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(llm_concurrency, "increment_llm_usage_many", failing_increment)
    with caplog.at_level(logging.ERROR, logger="api.llm_concurrency"):
        await asyncio.to_thread(_fake_llm().invoke, "hi", {"callbacks": [charger]})
        for _ in range(20):
            if caplog.records:
                break
            await asyncio.sleep(0.01)
    assert any("database is locked" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Price-weighted charging: one unit is one default-model call
# ---------------------------------------------------------------------------


def _blended(model) -> float | None:
    if model.input_price_per_mtok is None or model.output_price_per_mtok is None:
        return None
    return model.input_price_per_mtok + model.output_price_per_mtok


def budget_units(model) -> int:
    """Units one call on `model` should cost, from models.json prices:
    its blended (input + output) price over the default model's, rounded up."""
    price, base = _blended(model), _blended(get_default_model())
    if price is None or not base:
        return 1
    return max(1, math.ceil(round(price / base, 9)))


def pricier_lendable_model():
    """A model the operator lends whose call costs more than the default's."""
    pricier = [m for m in models_where(operator_lendable=True) if budget_units(m) > 1]
    if not pricier:
        pytest.skip("no operator_lendable model priced above the default model")
    return max(pricier, key=budget_units)


class _NamedFakeChatModel(FakeListChatModel):
    """Reports `model` as its model name, as real provider chat models do."""

    model: str


def _named_llm(model_id: str) -> FakeListChatModel:
    return _NamedFakeChatModel(model=model_id, responses=["ok"])


@pytest.mark.eval_unit
def test_default_model_call_costs_one_unit():
    assert llm_concurrency.call_units(get_default_model().id) == 1


@pytest.mark.eval_unit
@pytest.mark.parametrize("model", models_where(), ids=lambda m: m.id)
def test_call_units_follow_registry_prices(model):
    assert llm_concurrency.call_units(model.id) == budget_units(model)


@pytest.mark.eval_unit
def test_unknown_or_unpriced_model_costs_one_unit_with_a_warning(caplog):
    unpriced = [m.id for m in models_where() if _blended(m) is None]
    with caplog.at_level(logging.WARNING, logger="api.llm_concurrency"):
        for model_id in ["unknown-model", *unpriced]:
            assert llm_concurrency.call_units(model_id) == 1
    assert any("unknown-model" in r.getMessage() for r in caplog.records)


@pytest.mark.eval_unit
async def test_charger_weights_each_call_by_the_called_models_price(fresh_db):
    model = pricier_lendable_model()
    llm = _named_llm(model.id)
    charger = LLMCallCharger(USER_A, used=NONE_USED)
    for _ in range(2):
        await llm.ainvoke("hi", config={"callbacks": [charger]})
    assert await charger.settle() == 2
    assert await db_module.get_llm_usage(USER_A) == 2 * budget_units(model)


@pytest.mark.eval_unit
async def test_charger_refuses_a_call_the_budget_cannot_cover(fresh_db, monkeypatch):
    """With fewer units left than a pricey call costs, that call is refused,
    while a default-model call still fits."""
    model = pricier_lendable_model()
    units = budget_units(model)
    monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2 * units - 1)
    charger = LLMCallCharger(USER_A, used=NONE_USED)
    config = {"callbacks": [charger]}

    await _named_llm(model.id).ainvoke("hi", config=config)
    await charger.settle()
    with pytest.raises(LLMBudgetExceeded):
        await _named_llm(model.id).ainvoke("hi", config=config)
    await _named_llm(get_default_model().id).ainvoke("hi", config=config)

    assert await charger.settle() == 2
    assert await db_module.get_llm_usage(USER_A) == units + 1


@pytest.mark.eval_unit
@pytest.mark.parametrize("model", one_per_provider(), ids=lambda m: m.id)
def test_provider_chat_models_report_their_registry_id(model):
    """Weighting reads the model name each provider's chat model reports;
    it must be the registry id `create_llm` was given."""
    from agents.llm_factory import create_llm

    llm = create_llm(model.id, "not-a-real-key")
    reported = llm._get_ls_params_with_defaults()
    assert llm_concurrency._called_model_id({"metadata": reported}) == model.id
