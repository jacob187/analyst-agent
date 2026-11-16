"""Tests for watchlist REST endpoints."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.db as db_module
from agents.model_registry import get_default_model
from api import dependencies, llm_concurrency
from api.main import app
from tests.conftest import byok_header, model_where, set_operator_key


client = TestClient(app)


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Routes that reach the DB (e.g. ticker enrichment) must not open data/chats.db."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "default.db")
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.run(db_module.init_db())
    yield
    asyncio.run(db_module.close_db())


USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
HEADERS = {"X-User-Id": USER_ID}


class TestListWatchlist:
    @patch("api.routes.watchlist.get_watchlist_enriched", new_callable=AsyncMock)
    def test_list_empty(self, mock_get):
        mock_get.return_value = []
        resp = client.get("/watchlist", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json() == {"tickers": []}
        mock_get.assert_called_once_with(USER_ID)

    @patch("api.routes.watchlist.get_watchlist_enriched", new_callable=AsyncMock)
    def test_list_with_tickers(self, mock_get):
        mock_get.return_value = [
            {"ticker": "AAPL", "added_at": "2026-01-01", "name": "Apple", "sector": "Tech"},
        ]
        resp = client.get("/watchlist", headers=HEADERS)
        assert resp.status_code == 200
        assert len(resp.json()["tickers"]) == 1

    def test_list_without_user_id_returns_422(self):
        resp = client.get("/watchlist")
        assert resp.status_code == 422


class TestAddTicker:
    @patch("api.routes.watchlist.add_to_watchlist", new_callable=AsyncMock)
    def test_add_valid(self, mock_add):
        mock_add.return_value = True
        resp = client.post("/watchlist", json={"ticker": "AAPL"}, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"
        mock_add.assert_called_once_with("AAPL", USER_ID)

    def test_add_invalid_ticker(self):
        resp = client.post("/watchlist", json={"ticker": "!!!"}, headers=HEADERS)
        assert resp.status_code == 422

    @patch("api.routes.watchlist.add_to_watchlist", new_callable=AsyncMock)
    def test_add_duplicate_or_full(self, mock_add):
        mock_add.return_value = False
        resp = client.post("/watchlist", json={"ticker": "AAPL"}, headers=HEADERS)
        assert resp.status_code == 409

    def test_add_without_user_id_returns_422(self):
        resp = client.post("/watchlist", json={"ticker": "AAPL"})
        assert resp.status_code == 422


class TestRemoveTicker:
    @patch("api.routes.watchlist.remove_from_watchlist", new_callable=AsyncMock)
    def test_remove_existing(self, mock_remove):
        mock_remove.return_value = True
        resp = client.delete("/watchlist/AAPL", headers=HEADERS)
        assert resp.status_code == 200
        mock_remove.assert_called_once_with("AAPL", USER_ID)

    @patch("api.routes.watchlist.remove_from_watchlist", new_callable=AsyncMock)
    def test_remove_nonexistent(self, mock_remove):
        mock_remove.return_value = False
        resp = client.delete("/watchlist/XYZ", headers=HEADERS)
        assert resp.status_code == 404


class TestBriefingEndpoint:
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_empty_watchlist(self, mock_wl):
        mock_wl.return_value = []
        resp = client.get(
            "/watchlist/briefing",
            headers={**HEADERS, "X-Google-Api-Key": "test"}
        )
        assert resp.status_code == 400

    @patch.dict("os.environ", {}, clear=True)
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_no_api_key(self, mock_wl):
        mock_wl.return_value = [{"ticker": "AAPL", "added_at": "2026-01-01"}]
        resp = client.get("/watchlist/briefing", headers=HEADERS)
        assert resp.status_code == 400


class TestBriefingTimeout:
    @patch("api.routes.watchlist.get_recent_briefings", new_callable=AsyncMock)
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_briefing_timeout_returns_504(self, mock_wl, mock_recent, monkeypatch):
        """A stuck BriefingService.generate must surface as HTTP 504.

        Note: `asyncio.to_thread` wraps `time.sleep` in a thread Python cannot
        cancel, so the route's `wait_for` returns promptly to the user but the
        thread keeps running until its sleep ends. The TestClient (sync, via
        anyio portal) blocks until that thread drains, so wall-clock elapsed
        here will match the stub sleep duration. The user-visible 504 still
        fires at the timeout boundary — verified by the WARNING log line.
        """
        mock_wl.return_value = [{"ticker": "AAPL", "added_at": "2026-01-01"}]
        mock_recent.return_value = []  # no prior briefing to diff against

        # Compress timeout below the stub sleep so the wait_for path fires.
        monkeypatch.setattr("api.routes.watchlist.BRIEFING_TIMEOUT_SECONDS", 0.1)

        class _SlowService:
            def __init__(self, llm, tavily_api_key=None):
                pass

            def generate(self, tickers, user_id, model_id, previous_briefing=None, config=None):
                time.sleep(0.5)  # > timeout, < test budget
                return None

        fake_service_module = MagicMock()
        fake_service_module.BriefingService = _SlowService

        fake_llm_factory = MagicMock()
        fake_llm_factory.create_llm = MagicMock(return_value=object())
        fake_llm_factory.ThinkingConfig = MagicMock()

        import sys
        monkeypatch.setitem(sys.modules, "agents.briefing.briefing_service", fake_service_module)
        monkeypatch.setitem(sys.modules, "agents.llm_factory", fake_llm_factory)

        resp = client.get(
            "/watchlist/briefing",
            headers={**HEADERS, **byok_header(get_default_model().provider)},
        )

        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()

    @patch("api.routes.watchlist.get_recent_briefings", new_callable=AsyncMock)
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_briefing_success_under_timeout(self, mock_wl, mock_recent, monkeypatch):
        """Fast-returning generate should NOT be 504-ed."""
        mock_wl.return_value = [{"ticker": "AAPL", "added_at": "2026-01-01"}]
        mock_recent.return_value = []
        monkeypatch.setattr("api.routes.watchlist.BRIEFING_TIMEOUT_SECONDS", 2.0)

        class _FastService:
            def __init__(self, llm, tavily_api_key=None):
                pass

            def generate(self, tickers, user_id, model_id, previous_briefing=None, config=None):
                # Mimic the BriefingResult shape minimally for the persist path.
                analysis = MagicMock()
                analysis.model_dump_json.return_value = "{}"
                analysis.market_regime = "neutral"
                analysis.market_positioning = "balanced"
                analysis.alerts = []
                analysis.tickers = []
                result = MagicMock()
                result.analysis = analysis
                result.thinking = None
                return result

        fake_service_module = MagicMock()
        fake_service_module.BriefingService = _FastService

        fake_llm_factory = MagicMock()
        fake_llm_factory.create_llm = MagicMock(return_value=object())
        fake_llm_factory.ThinkingConfig = MagicMock()

        import sys
        monkeypatch.setitem(sys.modules, "agents.briefing.briefing_service", fake_service_module)
        monkeypatch.setitem(sys.modules, "agents.llm_factory", fake_llm_factory)

        # Also patch save_briefing so the success path doesn't touch DB.
        with patch("api.routes.watchlist.save_briefing", new_callable=AsyncMock):
            resp = client.get(
                "/watchlist/briefing",
                headers={**HEADERS, **byok_header(get_default_model().provider)},
            )
        assert resp.status_code == 200


class TestBriefingRateLimit:
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_briefing_rate_limited_after_5_calls(self, mock_wl):
        """6th briefing request from the same user within an hour returns 429.

        We let the first five calls fall through to the "empty watchlist"
        400, which is fine — the rate limiter gate runs before that branch,
        so each call still consumes quota.
        """
        mock_wl.return_value = []
        headers = {**HEADERS, "X-Google-Api-Key": "test"}
        for _ in range(5):
            resp = client.get("/watchlist/briefing", headers=headers)
            assert resp.status_code == 400  # empty watchlist, gate passed
        resp = client.get("/watchlist/briefing", headers=headers)
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()


class TestBriefingHistory:
    @patch("api.routes.watchlist.get_recent_briefings", new_callable=AsyncMock)
    def test_history_returns_list(self, mock_recent):
        mock_recent.return_value = [
            {"id": "abc", "market_regime": "Bull", "tickers": [], "created_at": "2026-04-01"},
        ]
        resp = client.get("/watchlist/briefing/history", headers=HEADERS)
        assert resp.status_code == 200
        assert len(resp.json()["briefings"]) == 1

    @patch("api.routes.watchlist.get_recent_briefings", new_callable=AsyncMock)
    def test_history_empty(self, mock_recent):
        mock_recent.return_value = []
        resp = client.get("/watchlist/briefing/history", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["briefings"] == []

    @patch("api.routes.watchlist.get_briefing_history", new_callable=AsyncMock)
    def test_history_by_ticker(self, mock_hist):
        mock_hist.return_value = [
            {"briefing_id": "abc", "outlook": "bullish", "price": 180.0, "created_at": "2026-04-01"},
        ]
        resp = client.get("/watchlist/briefing/history/AAPL", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"
        assert len(resp.json()["history"]) == 1

    @patch("api.routes.watchlist.get_briefing_history", new_callable=AsyncMock)
    def test_history_by_ticker_with_days_param(self, mock_hist):
        mock_hist.return_value = []
        resp = client.get("/watchlist/briefing/history/XOM?days=7", headers=HEADERS)
        assert resp.status_code == 200
        mock_hist.assert_called_once_with("XOM", USER_ID, days=7)

    def test_history_without_user_id_returns_422(self):
        resp = client.get("/watchlist/briefing/history")
        assert resp.status_code == 422


class TestBriefingHistoryDaysValidation:
    """`days` is range-bounded [1, 365] so unbounded inputs can't reach the DB."""

    def test_days_zero_rejected(self):
        resp = client.get("/watchlist/briefing/history/AAPL?days=0", headers=HEADERS)
        assert resp.status_code == 422

    def test_days_too_large_rejected(self):
        resp = client.get(
            "/watchlist/briefing/history/AAPL?days=999999", headers=HEADERS
        )
        assert resp.status_code == 422

    def test_days_negative_rejected(self):
        resp = client.get("/watchlist/briefing/history/AAPL?days=-1", headers=HEADERS)
        assert resp.status_code == 422

    @patch("api.routes.watchlist.get_briefing_history", new_callable=AsyncMock)
    def test_days_one_accepted(self, mock_hist):
        mock_hist.return_value = []
        resp = client.get(
            "/watchlist/briefing/history/AAPL?days=1", headers=HEADERS
        )
        assert resp.status_code == 200
        mock_hist.assert_called_once_with("AAPL", USER_ID, days=1)

    @patch("api.routes.watchlist.get_briefing_history", new_callable=AsyncMock)
    def test_days_365_accepted(self, mock_hist):
        mock_hist.return_value = []
        resp = client.get(
            "/watchlist/briefing/history/AAPL?days=365", headers=HEADERS
        )
        assert resp.status_code == 200
        mock_hist.assert_called_once_with("AAPL", USER_ID, days=365)


# ---------------------------------------------------------------------------
# Operator-key cost guard: lendable models only, one budget unit per LLM call
# ---------------------------------------------------------------------------

CLERK_USER = "user_abc123"
SIGNED_IN = {"X-Clerk-Session-Token": "good-token"}


class _CountingFakeLLM:
    """Records how many times the LLM boundary was invoked."""

    def __init__(self):
        self.created = 0


@pytest.fixture
def budget_env(monkeypatch):
    """Temp DB (the autouse `_temp_db`), a one-ticker watchlist, and a fake LLM at the provider boundary.

    The real BriefingService and route run; only the LLM (FakeListChatModel)
    and the yfinance/Tavily/regime fetches are faked.
    """
    from langchain_core.language_models import FakeListChatModel

    import agents.briefing.briefing_service as briefing_mod
    import agents.llm_factory as llm_factory
    from agents.briefing.briefing_service import (
        BriefingService,
        DailyBriefingAnalysis,
        TickerBriefing,
    )

    analysis = DailyBriefingAnalysis(
        market_regime="Neutral",
        market_positioning="Balanced.",
        tickers=[TickerBriefing(
            ticker="AAPL", price=180.0, change_pct=0.5,
            technical_signal="RSI neutral.", outlook="mixed",
        )],
        alerts=[],
    )
    fake = _CountingFakeLLM()

    def _create_llm(model_id, api_key, thinking=None):
        fake.created += 1
        return FakeListChatModel(responses=[analysis.model_dump_json()] * 5)

    monkeypatch.setattr(llm_factory, "create_llm", _create_llm)
    monkeypatch.setattr(
        BriefingService, "_gather_ticker_data",
        lambda self, tickers: [{"ticker": t, "price": 180.0} for t in tickers],
    )
    monkeypatch.setattr(BriefingService, "_get_market_regime", lambda self: {"error": "skip"})
    monkeypatch.setattr(BriefingService, "_gather_news", lambda self, tickers: {t: [] for t in tickers})
    briefing_mod._briefing_cache.clear()

    for uid in (CLERK_USER, USER_ID):
        asyncio.run(db_module.add_to_watchlist("AAPL", uid))

    yield fake
    briefing_mod._briefing_cache.clear()
    asyncio.run(db_module.close_db())


@pytest.fixture
def clerk_on(monkeypatch):
    """Hosted deploy: Clerk enabled, JWT verification faked at its boundary."""
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setattr(
        dependencies, "verify_clerk_token",
        lambda token: CLERK_USER if token == "good-token" else None,
    )
    dependencies._clerk_unconfigured_warned = False


def _usage(user_id: str) -> int:
    return asyncio.run(db_module.get_llm_usage(user_id))


@pytest.fixture
def lendable():
    return model_where(operator_lendable=True)


@pytest.fixture
def non_lendable():
    return model_where(operator_lendable=False)


class TestBriefingOperatorKeyGuard:
    @pytest.fixture(autouse=True)
    def _operator_env(self, monkeypatch, lendable, non_lendable):
        set_operator_key(monkeypatch, lendable, "operator-key")
        set_operator_key(monkeypatch, non_lendable, "operator-key")

    @pytest.mark.eval_unit
    def test_lendable_model_on_operator_key_charges_one_unit_per_call(self, budget_env, clerk_on, lendable):
        resp = client.get("/watchlist/briefing", headers={**SIGNED_IN, "X-Model-Id": lendable.id})
        assert resp.status_code == 200
        assert _usage(CLERK_USER) == 1  # one LLM call in the briefing chain

    @pytest.mark.eval_unit
    def test_cached_briefing_makes_no_llm_call_and_is_not_charged(self, budget_env, clerk_on, lendable):
        headers = {**SIGNED_IN, "X-Model-Id": lendable.id}
        assert client.get("/watchlist/briefing", headers=headers).status_code == 200
        assert client.get("/watchlist/briefing", headers=headers).status_code == 200
        assert _usage(CLERK_USER) == 1

    @pytest.mark.eval_unit
    def test_non_lendable_model_refused_on_operator_key(self, budget_env, clerk_on, non_lendable):
        resp = client.get("/watchlist/briefing", headers={**SIGNED_IN, "X-Model-Id": non_lendable.id})
        assert resp.status_code == 400
        assert "own" in resp.json()["detail"].lower()
        assert budget_env.created == 0  # never reached the LLM
        assert _usage(CLERK_USER) == 0

    @pytest.mark.eval_unit
    def test_byok_works_for_non_lendable_model_and_is_not_charged(self, budget_env, clerk_on, non_lendable):
        resp = client.get(
            "/watchlist/briefing",
            headers={**SIGNED_IN, "X-Model-Id": non_lendable.id, **byok_header(non_lendable.provider, "mine")},
        )
        assert resp.status_code == 200
        assert budget_env.created == 1
        assert _usage(CLERK_USER) == 0

    @pytest.mark.eval_unit
    def test_budget_spent_refuses_before_any_llm_call(self, budget_env, clerk_on, monkeypatch, lendable):
        monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 1)
        asyncio.run(db_module.increment_llm_usage(CLERK_USER))
        resp = client.get("/watchlist/briefing", headers={**SIGNED_IN, "X-Model-Id": lendable.id})
        assert resp.status_code == 429
        assert budget_env.created == 0
        assert _usage(CLERK_USER) == 1

    @pytest.mark.eval_unit
    def test_byok_ignores_spent_budget(self, budget_env, clerk_on, monkeypatch, lendable):
        monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 1)
        asyncio.run(db_module.increment_llm_usage(CLERK_USER))
        resp = client.get(
            "/watchlist/briefing",
            headers={**SIGNED_IN, "X-Model-Id": lendable.id, **byok_header(lendable.provider, "mine")},
        )
        assert resp.status_code == 200
        assert _usage(CLERK_USER) == 1

    @pytest.mark.eval_unit
    def test_generation_failure_detail_is_generic_and_logged(
        self, budget_env, monkeypatch, lendable, caplog
    ):
        from agents.briefing.briefing_service import BriefingService

        def _boom(self, tickers):
            raise RuntimeError("upstream rejected key tvly-secret-123")

        monkeypatch.setattr(BriefingService, "_gather_news", _boom)
        with caplog.at_level("ERROR", logger="analyst.watchlist"):
            resp = client.get(
                "/watchlist/briefing",
                headers={**HEADERS, "X-Model-Id": lendable.id, **byok_header(lendable.provider)},
            )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Briefing generation failed"
        assert "tvly-secret-123" not in resp.text
        assert any(r.exc_info and "tvly-secret-123" in str(r.exc_info[1]) for r in caplog.records)

    @pytest.mark.eval_unit
    def test_self_host_lends_operator_key_for_any_model(self, budget_env, non_lendable):
        # Clerk unconfigured: the operator is the user, every model is lendable.
        resp = client.get("/watchlist/briefing", headers={**HEADERS, "X-Model-Id": non_lendable.id})
        assert resp.status_code == 200
        assert _usage(USER_ID) == 1
