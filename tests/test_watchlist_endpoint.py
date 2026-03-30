"""Tests for watchlist REST endpoints."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


class TestListWatchlist:
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_list_empty(self, mock_get):
        mock_get.return_value = []
        resp = client.get("/watchlist")
        assert resp.status_code == 200
        assert resp.json() == {"tickers": []}

    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_list_with_tickers(self, mock_get):
        mock_get.return_value = [
            {"ticker": "AAPL", "added_at": "2026-01-01"},
            {"ticker": "MSFT", "added_at": "2026-01-02"},
        ]
        resp = client.get("/watchlist")
        assert resp.status_code == 200
        assert len(resp.json()["tickers"]) == 2


class TestAddTicker:
    @patch("api.routes.watchlist.add_to_watchlist", new_callable=AsyncMock)
    def test_add_valid(self, mock_add):
        mock_add.return_value = True
        resp = client.post("/watchlist", json={"ticker": "AAPL"})
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "AAPL"

    def test_add_invalid_ticker(self):
        resp = client.post("/watchlist", json={"ticker": "!!!"})
        assert resp.status_code == 422

    @patch("api.routes.watchlist.add_to_watchlist", new_callable=AsyncMock)
    def test_add_duplicate_or_full(self, mock_add):
        mock_add.return_value = False
        resp = client.post("/watchlist", json={"ticker": "AAPL"})
        assert resp.status_code == 409


class TestRemoveTicker:
    @patch("api.routes.watchlist.remove_from_watchlist", new_callable=AsyncMock)
    def test_remove_existing(self, mock_remove):
        mock_remove.return_value = True
        resp = client.delete("/watchlist/AAPL")
        assert resp.status_code == 200

    @patch("api.routes.watchlist.remove_from_watchlist", new_callable=AsyncMock)
    def test_remove_nonexistent(self, mock_remove):
        mock_remove.return_value = False
        resp = client.delete("/watchlist/XYZ")
        assert resp.status_code == 404


class TestBriefingEndpoint:
    @patch("api.routes.watchlist.get_settings", new_callable=AsyncMock)
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_empty_watchlist(self, mock_wl, mock_settings):
        mock_wl.return_value = []
        resp = client.get("/watchlist/briefing")
        assert resp.status_code == 400

    @patch("api.routes.watchlist.get_settings", new_callable=AsyncMock)
    @patch("api.routes.watchlist.get_watchlist", new_callable=AsyncMock)
    def test_no_api_key(self, mock_wl, mock_settings):
        mock_wl.return_value = [{"ticker": "AAPL", "added_at": "2026-01-01"}]
        mock_settings.return_value = None
        resp = client.get("/watchlist/briefing")
        assert resp.status_code == 400
