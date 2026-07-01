"""Tests for the /api/company/{ticker}/filings endpoint.

Mocks SEC data retrieval, LLM analysis, and DB layer so tests run
without network access, API keys, or LLM calls.
"""

import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Shared mock data ───────────────────────────────────────────────────────

_MOCK_TENK_METADATA = {
    "form": "10-K",
    "cik": "320193",
    "accession": "0000320193-24-000123",
    "filing_date": "2025-11-01",
    "period_of_report": "2025-09-30",
    "company_name": "APPLE INC",
}

_MOCK_TENQ_METADATA = {
    "form": "10-Q",
    "cik": "320193",
    "accession": "0000320193-25-000456",
    "filing_date": "2026-02-01",
    "period_of_report": "2025-12-31",
    "company_name": "APPLE INC",
}

_MOCK_8K_METADATA = {
    "form": "8-K",
    "cik": "320193",
    "accession": "0000320193-26-000789",
    "filing_date": "2026-01-30",
    "period_of_report": "2026-01-30",
    "company_name": "APPLE INC",
}

_MOCK_RISK_ANALYSIS = {
    "summary": "Key risks include supply chain and regulation.",
    "key_risks": ["Supply chain disruption", "Regulatory changes"],
    "risk_categories": {"Operational": ["Supply chain"]},
    "sentiment_score": -3.0,
    "sentiment_analysis": "Moderately negative risk profile.",
    "form_type": "10-K",
    "filing_metadata": _MOCK_TENK_METADATA,
    "comparison": None,
}

_MOCK_MDA_ANALYSIS = {
    "summary": "Revenue grew 8% YoY driven by Services.",
    "key_points": ["Services revenue up 15%"],
    "financial_highlights": ["Revenue: $394B"],
    "future_outlook": "Positive with AI investments.",
    "sentiment_score": 6.0,
    "sentiment_analysis": "Positive tone.",
    "form_type": "10-K",
    "filing_metadata": _MOCK_TENK_METADATA,
    "comparison": None,
}

_MOCK_BALANCE_ANALYSIS = {
    "summary": "Strong balance sheet.",
    "key_metrics": ["Total assets: $352B"],
    "liquidity_analysis": "Good liquidity.",
    "solvency_analysis": "Low leverage.",
    "growth_trends": "Steady growth.",
    "financial_highlights": ["Cash: $62B"],
    "red_flags": [],
    "comparison": None,
}

_MOCK_EARNINGS_ANALYSIS = {
    "summary": "Revenue beat estimates by 3%.",
    "key_metrics": ["Revenue: $94.9B", "EPS: $2.40"],
    "beats_misses": ["Revenue beat by $2.8B"],
    "guidance": "Q2 revenue $90-92B",
    "sentiment_score": 7.5,
    "sentiment_analysis": "Strong results.",
    "filing_metadata": _MOCK_8K_METADATA,
}

_MOCK_EVENT_ANALYSIS = {
    "summary": "CEO transition announced.",
    "event_type": "Leadership Change",
    "key_points": ["CEO stepping down effective immediately", "CFO named interim CEO"],
    "impact_assessment": "Short-term volatility expected; long-term impact depends on successor.",
    "sentiment_score": -2.0,
    "sentiment_analysis": "Mildly negative pending clarity on succession.",
    "filing_metadata": _MOCK_8K_METADATA,
}


# ── Fixture that mocks all dependencies ───────────────────────────────────

@pytest.fixture
def mock_filings_deps():
    """Patch SEC retrieval, LLM analysis, DB, and model registry."""
    # Mock the FilingMetadata objects used inside _fetch_filing_data
    mock_tenk_meta = MagicMock()
    mock_tenk_meta.to_dict.return_value = _MOCK_TENK_METADATA
    mock_tenk_meta.cik = "320193"
    mock_tenk_meta.accession = "0000320193-24-000123"

    mock_tenq_meta = MagicMock()
    mock_tenq_meta.to_dict.return_value = _MOCK_TENQ_METADATA
    mock_tenq_meta.cik = "320193"
    mock_tenq_meta.accession = "0000320193-25-000456"

    mock_8k_meta = MagicMock()
    mock_8k_meta.to_dict.return_value = _MOCK_8K_METADATA
    mock_8k_meta.cik = "320193"
    mock_8k_meta.accession = "0000320193-26-000789"

    with (
        patch("agents.sec_workflow.get_SEC_data.SECDataRetrieval") as MockSEC,
        patch("agents.llm_factory.create_llm") as MockCreateLLM,
        patch("agents.sec_workflow.sec_llm_models.SECDocumentProcessor") as MockProcessor,
        patch("api.db.get_filing_analysis", new_callable=AsyncMock) as mock_get_cache,
        patch("api.db.save_filing_analysis", new_callable=AsyncMock) as mock_save_cache,
        patch("agents.model_registry.get_model") as mock_get_model,
        patch("agents.model_registry.get_default_model") as mock_default_model,
    ):
        # Model registry
        model = MagicMock()
        model.id = "gemini-2.5-flash"
        model.provider = "google_genai"
        mock_get_model.return_value = model
        mock_default_model.return_value = model

        # SEC retriever instance
        sec_instance = MagicMock()
        sec_instance._tenk_metadata = mock_tenk_meta
        sec_instance._tenq_metadata = mock_tenq_meta
        sec_instance._eightk_metadata = mock_8k_meta
        sec_instance.get_tenk_filing.return_value = MagicMock()
        sec_instance.get_tenq_filing.return_value = MagicMock()
        sec_instance.get_risk_factors_raw.return_value = {
            "text": "Risk factor text...", "metadata": _MOCK_TENK_METADATA, "found": True,
        }
        sec_instance.get_mda_raw.return_value = {
            "text": "MD&A text...", "metadata": _MOCK_TENK_METADATA, "found": True,
        }
        sec_instance.extract_balance_sheet_as_str.return_value = {"10-K": "Balance sheet text"}
        sec_instance.get_8k_overview.return_value = {
            "found": True,
            "has_earnings": True,
            "items": ["2.02"],
            "content_type": "earnings",
            "is_amendment": False,
            "has_press_release": True,
            "date_of_report": "2026-01-30",
            "context": "Q1 2026 earnings context",
            "metadata": _MOCK_8K_METADATA,
        }
        sec_instance.get_8k_item.return_value = {
            "text": "Item text...", "metadata": _MOCK_8K_METADATA, "found": True,
        }
        sec_instance.get_earnings_data.return_value = {
            "has_earnings": True,
            "context": "Q1 2026 earnings context",
            "detected_scale": "millions",
            "metadata": _MOCK_8K_METADATA,
        }
        MockSEC.return_value = sec_instance

        # LLM + processor
        mock_llm = MagicMock()
        MockCreateLLM.return_value = mock_llm

        processor_instance = MagicMock()
        processor_instance.analyze_risk_factors.return_value = MagicMock(
            model_dump=lambda: _MOCK_RISK_ANALYSIS
        )
        processor_instance.analyze_mda.return_value = MagicMock(
            model_dump=lambda: _MOCK_MDA_ANALYSIS
        )
        processor_instance.analyze_balance_sheet.return_value = MagicMock(
            model_dump=lambda: _MOCK_BALANCE_ANALYSIS
        )
        processor_instance.analyze_earnings.return_value = MagicMock(
            model_dump=lambda: _MOCK_EARNINGS_ANALYSIS
        )
        processor_instance.analyze_material_event.return_value = MagicMock(
            model_dump=lambda: _MOCK_EVENT_ANALYSIS
        )
        MockProcessor.return_value = processor_instance

        # DB cache — default to cache miss (None)
        mock_get_cache.return_value = None
        mock_save_cache.return_value = "analysis-id-123"

        yield {
            "sec": sec_instance,
            "processor": processor_instance,
            "get_cache": mock_get_cache,
            "save_cache": mock_save_cache,
            "model": model,
        }


# ── Happy path ─────────────────────────────────────────────────────────────


class TestFilingsEndpointSuccess:
    def test_returns_200_with_all_sections(self, client, mock_filings_deps):
        resp = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert "tenk" in data
        assert "tenq" in data
        assert "eightk" in data

    def test_tenk_has_all_analysis_types(self, client, mock_filings_deps):
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        ).json()
        tenk = data["tenk"]
        assert "metadata" in tenk
        assert "risk_10k" in tenk
        assert "mda_10k" in tenk
        assert "balance" in tenk
        assert tenk["risk_10k"]["summary"] == "Key risks include supply chain and regulation."

    def test_earnings_section_when_present(self, client, mock_filings_deps):
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        ).json()
        eightk = data["eightk"]
        assert eightk["kind"] == "earnings"
        assert "analysis" in eightk
        assert eightk["analysis"]["summary"] == "Revenue beat estimates by 3%."

    def test_edgar_url_construction(self, client, mock_filings_deps):
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        ).json()
        edgar_url = data["tenk"]["metadata"]["edgar_url"]
        assert "sec.gov/Archives/edgar/data/320193/" in edgar_url
        # Accession number hyphens should be stripped in the URL
        assert "-" not in edgar_url.split("/data/320193/")[1]


# ── DB caching ─────────────────────────────────────────────────────────────


class TestFilingsCaching:
    def test_cache_miss_calls_llm_and_saves(self, client, mock_filings_deps):
        """On cache miss, LLM analysis should run and result saved to DB."""
        client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        )
        # save_filing_analysis should have been called for each analysis type
        save_calls = mock_filings_deps["save_cache"].call_args_list
        analysis_types = [call.args[3] for call in save_calls]
        assert "risk_10k" in analysis_types
        assert "mda_10k" in analysis_types
        assert "balance" in analysis_types
        assert "earnings" in analysis_types

    def test_cache_hit_skips_llm(self, client, mock_filings_deps):
        """On cache hit, should return from DB without calling LLM."""
        mock_filings_deps["get_cache"].return_value = {
            "id": "cached-id",
            "analysis_json": json.dumps(_MOCK_RISK_ANALYSIS),
            "created_at": "2026-04-01T00:00:00",
        }
        resp = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        )
        assert resp.status_code == 200
        # Processor should not be called since everything is cached
        mock_filings_deps["processor"].analyze_risk_factors.assert_not_called()
        mock_filings_deps["processor"].analyze_mda.assert_not_called()
        mock_filings_deps["processor"].analyze_balance_sheet.assert_not_called()


# ── No earnings ────────────────────────────────────────────────────────────


class TestFilings8K:
    def test_no_eightk_when_overview_not_found(self, client, mock_filings_deps):
        mock_filings_deps["sec"].get_8k_overview.return_value = {
            "found": False,
            "text": "No 8-K available",
            "metadata": {},
        }
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        ).json()
        assert data["eightk"]["kind"] == "none"
        assert "reason" in data["eightk"]

    def test_material_event_when_non_earnings_eightk(self, client, mock_filings_deps):
        """Non-earnings 8-K (e.g. leadership change) should route to event analyzer."""
        mock_filings_deps["sec"].get_8k_overview.return_value = {
            "found": True,
            "has_earnings": False,
            "items": ["5.02"],
            "content_type": "director_change",
            "is_amendment": False,
            "has_press_release": False,
            "date_of_report": "2026-02-15",
            "context": "Departure of CEO; appointment of interim CEO.",
            "metadata": _MOCK_8K_METADATA,
        }
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        ).json()
        assert data["eightk"]["kind"] == "event"
        assert data["eightk"]["analysis"]["event_type"] == "Leadership Change"
        assert data["eightk"]["analysis"]["impact_assessment"].startswith("Short-term")
        # Event analyzer should have been invoked, earnings analyzer should not
        mock_filings_deps["processor"].analyze_material_event.assert_called_once()
        mock_filings_deps["processor"].analyze_earnings.assert_not_called()


# ── Error cases ────────────────────────────────────────────────────────────


class TestFilingsEndpointErrors:
    def test_invalid_ticker_returns_422(self, client):
        resp = client.get("/api/company/INVALID!!!/filings")
        assert resp.status_code == 422


# ── Anonymous access (Browse-only + BYOK) ──────────────────────────────────


class TestFilingsAnonAccess:
    """Cached analyses are public for discoverability; generating a fresh
    analysis on cache miss needs a key (BYOK, or operator key signed-in)."""

    def test_anon_with_byok_key_generates(self, client, mock_filings_deps):
        # No user_id, but a BYOK key → fresh analysis runs on cache miss.
        resp = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "byok-key"},
        )
        assert resp.status_code == 200
        assert "risk_10k" in resp.json()["tenk"]

    def test_anon_no_key_serves_cache(self, client, mock_filings_deps):
        # No user_id, no key, everything cached → public read, no LLM call.
        mock_filings_deps["get_cache"].return_value = {
            "analysis_json": json.dumps(_MOCK_RISK_ANALYSIS)
        }
        with patch.dict("os.environ", {}, clear=True):
            with patch("api.dependencies.os.getenv", return_value=None):
                resp = client.get("/api/company/AAPL/filings")
        assert resp.status_code == 200
        assert resp.json()["tenk"]["risk_10k"]["summary"] == _MOCK_RISK_ANALYSIS["summary"]
        mock_filings_deps["processor"].analyze_risk_factors.assert_not_called()

    def test_keyless_cache_miss_skips_llm(self, client, mock_filings_deps):
        # Keyless caller + cache miss → 200, but no analysis generated and no
        # LLM call. Holds whether anonymous or signed-in without a key.
        with patch.dict("os.environ", {}, clear=True):
            with patch("api.dependencies.os.getenv", return_value=None):
                resp = client.get(
                    "/api/company/AAPL/filings",
                    headers={"X-User-Id": "user_testfilings"},
                )
        assert resp.status_code == 200
        assert "risk_10k" not in resp.json().get("tenk", {})
        mock_filings_deps["processor"].analyze_risk_factors.assert_not_called()


# ── Rate limiting ─────────────────────────────────────────────────────────


class TestFilingsRateLimit:
    def test_filings_rate_limited_after_10_calls(self, client, mock_filings_deps):
        """11th request from the same caller within an hour returns 429."""
        headers = {"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"}
        for _ in range(10):
            resp = client.get("/api/company/AAPL/filings", headers=headers)
            assert resp.status_code == 200
        resp = client.get("/api/company/AAPL/filings", headers=headers)
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()


# ── Partial data ───────────────────────────────────────────────────────────


class TestFilingsPartialData:
    def test_no_tenk_returns_without_tenk_section(self, client, mock_filings_deps):
        """Companies without a 10-K or 20-F (e.g., new IPOs) should still return."""
        mock_filings_deps["sec"].get_tenk_filing.side_effect = ValueError("No 10-K")
        mock_filings_deps["sec"].get_twentyf_filing.side_effect = ValueError("No 20-F")
        data = client.get(
            "/api/company/NEWIPO/filings",
            headers={"X-Google-Api-Key": "test-key", "X-User-Id": "user_testfilings"},
        ).json()
        assert "tenk" not in data
        # 10-Q and earnings should still be present
        assert "tenq" in data


# ── Daily LLM budget (operator-paid only) ─────────────────────────────────


class TestFilingsLLMBudget:
    """The daily budget gate fires when the request's provider key comes from
    server env (operator pays). BYOK requests bypass it."""

    @pytest.fixture
    def budget_db(self, tmp_path, monkeypatch):
        import api.db as db_module
        monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "budget.db")
        monkeypatch.setattr(db_module, "_db", None)
        # init_db is async, but TestClient drives the FastAPI app which has
        # init_db in its lifespan. Force the connection open eagerly so
        # increment_llm_usage finds the table on the very first call.
        import asyncio as _asyncio
        _asyncio.get_event_loop().run_until_complete(db_module.init_db())
        yield
        _asyncio.get_event_loop().run_until_complete(db_module.close_db())

    def test_byok_bypasses_budget(self, client, mock_filings_deps, budget_db, monkeypatch):
        """Header-supplied key (BYOK) is not charged against the operator's budget."""
        from api import llm_concurrency
        monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 1)
        headers = {"X-Google-Api-Key": "user-key", "X-User-Id": "user_byoktest"}
        # Five calls all succeed — the 1-call budget would have blocked all
        # but the first if BYOK weren't bypassing.
        for _ in range(5):
            resp = client.get("/api/company/AAPL/filings", headers=headers)
            assert resp.status_code == 200

    def test_env_key_subject_to_budget(self, client, mock_filings_deps, budget_db, monkeypatch):
        """When the key resolves from env (no header), the daily budget applies."""
        from api import llm_concurrency
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        monkeypatch.setattr(llm_concurrency, "LLM_DAILY_BUDGET", 2)
        headers = {"X-User-Id": "user_envbudget"}
        # First two succeed; third crosses the limit and returns 429.
        assert client.get("/api/company/AAPL/filings", headers=headers).status_code == 200
        assert client.get("/api/company/AAPL/filings", headers=headers).status_code == 200
        resp = client.get("/api/company/AAPL/filings", headers=headers)
        assert resp.status_code == 429
        assert "budget" in resp.json()["detail"].lower()


# ── Process-wide LLM dispatch concurrency cap ────────────────────────────


class TestFilingsConcurrencyCap:
    async def test_filings_llm_concurrency_capped(self, mock_filings_deps, monkeypatch):
        """Concurrent /filings requests must never exceed LLM_DISPATCH_CONCURRENCY
        in-flight LLM dispatches process-wide.

        Drives the app via httpx.AsyncClient (TestClient is sync) so we can
        actually fire requests in parallel. Patches `_run_llm_analysis` with
        a sleep stub that tracks max in-flight.
        """
        import asyncio as _asyncio
        import threading
        import time as _time

        import httpx
        from api import llm_concurrency
        from api.routes import company as company_route

        monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_CONCURRENCY", 3)
        llm_concurrency._rebuild_semaphore()

        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def slow_analysis(ticker, analysis_type, raw_data, model_id, api_key):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            _time.sleep(0.05)
            with lock:
                in_flight -= 1
            return {"summary": "stub", "filing_metadata": {"accession": "x"}}

        monkeypatch.setattr(company_route, "_run_llm_analysis", slow_analysis)

        headers = {"X-Google-Api-Key": "test-key", "X-User-Id": "user_conccap"}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Fire 5 concurrent requests against distinct tickers (each
            # request also dispatches multiple sections internally; with
            # cap=3 the peak must never exceed 3 anywhere).
            await _asyncio.gather(*[
                ac.get(f"/api/company/T{i}/filings", headers=headers)
                for i in range(5)
            ])

        assert peak <= 3, f"observed peak {peak} > cap 3"
        assert peak >= 2, f"test setup didn't actually exercise concurrency (peak={peak})"

        # Restore the default semaphore size for the rest of the suite.
        monkeypatch.undo()
        llm_concurrency._rebuild_semaphore()


# ── Streaming endpoint (SSE) ───────────────────────────────────────────────


from contextlib import contextmanager


@contextmanager
def _no_keys():
    """Force a truly keyless request: the project .env may set GOOGLE_API_KEY,
    and with Clerk disabled the env key would otherwise resolve for an anon
    caller. Mirror the existing keyless tests' env-clearing pattern."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("api.dependencies.os.getenv", return_value=None):
            yield


def _stream_events(client, path, headers=None):
    """Open the SSE filings stream and collect parsed `data:` events."""
    events = []
    with client.stream("GET", path, headers=headers or {}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _core_only(deps):
    """Restrict the SEC mock to core sections so the stream's conditional
    (Item 1/1C/3/7A) and XBRL sections produce no tasks — keeps the event set
    predictable. Unconfigured MagicMock getters otherwise look 'found'."""
    sec = deps["sec"]
    for m in ("get_business_raw", "get_cybersecurity_raw",
              "get_legal_proceedings_raw", "get_market_risk_raw"):
        getattr(sec, m).return_value = {"found": False}
    sec.get_income_statement.return_value = None
    sec.get_cashflow_statement.return_value = None


_KEYED = {"X-Google-Api-Key": "test-key", "X-User-Id": "user_streamtest"}


class TestFilingsStream:
    """Characterization of GET /{ticker}/filings/stream (previously untested)."""

    def test_stream_generates_and_completes(self, client, mock_filings_deps):
        _core_only(mock_filings_deps)
        events = _stream_events(client, "/api/company/AAPL/filings/stream", _KEYED)
        types = [e.get("type") for e in events]
        assert "metadata" in types
        assert types[-1] == "complete"
        # Cache miss + key → core sections generate and stream a section payload.
        section_keys = {e["key"] for e in events if e.get("type") == "section"}
        assert {"risk_10k", "mda_10k", "balance"} <= section_keys
        statuses = {e["status"] for e in events if e.get("type") == "progress"}
        assert "done" in statuses

    def test_stream_serves_cached_without_llm(self, client, mock_filings_deps):
        _core_only(mock_filings_deps)
        mock_filings_deps["get_cache"].return_value = {
            "analysis_json": json.dumps(_MOCK_RISK_ANALYSIS)
        }
        events = _stream_events(client, "/api/company/AAPL/filings/stream", _KEYED)
        statuses = {e["status"] for e in events if e.get("type") == "progress"}
        assert "cached" in statuses
        assert "done" not in statuses  # nothing generated
        mock_filings_deps["processor"].analyze_risk_factors.assert_not_called()

    def test_stream_keyless_emits_needs_key(self, client, mock_filings_deps):
        _core_only(mock_filings_deps)
        with _no_keys():
            events = _stream_events(client, "/api/company/AAPL/filings/stream")
        statuses = {e["status"] for e in events if e.get("type") == "progress"}
        assert "needs_key" in statuses
        assert not any(e.get("type") == "section" for e in events)
        mock_filings_deps["processor"].analyze_risk_factors.assert_not_called()


# ── #5: keyless callers must not pay for raw section extraction ────────────


class TestKeylessSkipsRawExtraction:
    def test_stream_keyless_skips_raw_extraction(self, client, mock_filings_deps):
        _core_only(mock_filings_deps)
        with _no_keys():
            _stream_events(client, "/api/company/AAPL/filings/stream")
        sec = mock_filings_deps["sec"]
        sec.get_risk_factors_raw.assert_not_called()
        sec.get_mda_raw.assert_not_called()
        sec.extract_balance_sheet_as_str.assert_not_called()
        sec.get_earnings_data.assert_not_called()

    def test_get_keyless_skips_raw_extraction(self, client, mock_filings_deps):
        with _no_keys():
            client.get("/api/company/AAPL/filings")
        sec = mock_filings_deps["sec"]
        sec.get_risk_factors_raw.assert_not_called()
        sec.get_mda_raw.assert_not_called()
        sec.extract_balance_sheet_as_str.assert_not_called()


# ── #6: budget charged only when an LLM call actually happens ──────────────


class TestBudgetChargedOnlyOnGeneration:
    def test_fully_cached_does_not_charge_budget(self, client, mock_filings_deps, monkeypatch):
        from unittest.mock import AsyncMock as _AsyncMock
        from api.routes import company as company_route

        mock_charge = _AsyncMock()
        monkeypatch.setattr(company_route, "check_and_charge_budget", mock_charge)
        # Operator-paid (env key, no header) + every section cached.
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        mock_filings_deps["get_cache"].return_value = {
            "analysis_json": json.dumps(_MOCK_RISK_ANALYSIS)
        }
        resp = client.get("/api/company/AAPL/filings", headers={"X-User-Id": "user_cachedbudget"})
        assert resp.status_code == 200
        mock_charge.assert_not_called()

    def test_stream_fully_cached_does_not_charge_budget(self, client, mock_filings_deps, monkeypatch):
        from unittest.mock import AsyncMock as _AsyncMock
        from api.routes import company as company_route

        _core_only(mock_filings_deps)
        mock_charge = _AsyncMock()
        monkeypatch.setattr(company_route, "check_and_charge_budget", mock_charge)
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        mock_filings_deps["get_cache"].return_value = {
            "analysis_json": json.dumps(_MOCK_RISK_ANALYSIS)
        }
        events = _stream_events(
            client, "/api/company/AAPL/filings/stream", {"X-User-Id": "user_streamcached"}
        )
        assert events[-1]["type"] == "complete"
        mock_charge.assert_not_called()
