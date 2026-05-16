"""Tests for watchlist REST endpoints."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

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
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_briefing_timeout_returns_504(self, mock_wl, monkeypatch):
        """A stuck BriefingService.generate must surface as HTTP 504.

        Note: `asyncio.to_thread` wraps `time.sleep` in a thread Python cannot
        cancel, so the route's `wait_for` returns promptly to the user but the
        thread keeps running until its sleep ends. The TestClient (sync, via
        anyio portal) blocks until that thread drains, so wall-clock elapsed
        here will match the stub sleep duration. The user-visible 504 still
        fires at the timeout boundary — verified by the WARNING log line.
        """
        mock_wl.return_value = [{"ticker": "AAPL", "added_at": "2026-01-01"}]

        # Compress timeout below the stub sleep so the wait_for path fires.
        monkeypatch.setattr("api.routes.watchlist.BRIEFING_TIMEOUT_SECONDS", 0.1)

        class _SlowService:
            def __init__(self, llm, tavily_api_key=None):
                pass

            def generate(self, tickers):
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
            headers={**HEADERS, "X-Google-Api-Key": "test"},
        )

        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()

    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_briefing_success_under_timeout(self, mock_wl, monkeypatch):
        """Fast-returning generate should NOT be 504-ed."""
        mock_wl.return_value = [{"ticker": "AAPL", "added_at": "2026-01-01"}]
        monkeypatch.setattr("api.routes.watchlist.BRIEFING_TIMEOUT_SECONDS", 2.0)

        class _FastService:
            def __init__(self, llm, tavily_api_key=None):
                pass

            def generate(self, tickers):
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
                headers={**HEADERS, "X-Google-Api-Key": "test"},
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
