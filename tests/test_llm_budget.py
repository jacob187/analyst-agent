"""Per-user daily LLM budget tests.

Covers the DB primitive (`increment_llm_usage` / `get_llm_usage`) and the
end-to-end behavior through `check_and_charge_budget`. The HTTP-level
budget enforcement is exercised via the filings endpoint, which is the
canonical operator-paid LLM dispatch site.
"""

import pytest

import api.db as db_module
from api import llm_concurrency
from api.llm_concurrency import LLMBudgetExceeded, check_and_charge_budget


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
