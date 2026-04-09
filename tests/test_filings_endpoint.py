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
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert "tenk" in data
        assert "tenq" in data
        assert "earnings" in data

    def test_tenk_has_all_analysis_types(self, client, mock_filings_deps):
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
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
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
        ).json()
        earnings = data["earnings"]
        assert earnings["has_earnings"] is True
        assert "analysis" in earnings
        assert earnings["analysis"]["summary"] == "Revenue beat estimates by 3%."

    def test_edgar_url_construction(self, client, mock_filings_deps):
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
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
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
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
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
        )
        assert resp.status_code == 200
        # Processor should not be called since everything is cached
        mock_filings_deps["processor"].analyze_risk_factors.assert_not_called()
        mock_filings_deps["processor"].analyze_mda.assert_not_called()
        mock_filings_deps["processor"].analyze_balance_sheet.assert_not_called()


# ── No earnings ────────────────────────────────────────────────────────────


class TestFilingsNoEarnings:
    def test_no_earnings_section_when_absent(self, client, mock_filings_deps):
        mock_filings_deps["sec"].get_earnings_data.return_value = {
            "has_earnings": False,
            "reason": "Latest 8-K does not contain Item 2.02",
        }
        data = client.get(
            "/api/company/AAPL/filings",
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
        ).json()
        assert data["earnings"]["has_earnings"] is False
        assert "reason" in data["earnings"]


# ── Error cases ────────────────────────────────────────────────────────────


class TestFilingsEndpointErrors:
    def test_missing_api_key_returns_400(self, client, mock_filings_deps):
        # No API key header and mock env to be empty
        with patch.dict("os.environ", {}, clear=True):
            with patch("api.dependencies.os.getenv", return_value=None):
                resp = client.get(
                    "/api/company/AAPL/filings",
                    headers={"X-Sec-Header": "test@test.com"},
                )
                assert resp.status_code == 400
                assert "API key required" in resp.json()["detail"]

    def test_missing_sec_header_returns_400(self, client, mock_filings_deps):
        with patch.dict("os.environ", {}, clear=True):
            with patch("api.dependencies.os.getenv", return_value=None):
                resp = client.get(
                    "/api/company/AAPL/filings",
                    headers={"X-Google-Api-Key": "test-key"},
                )
                assert resp.status_code == 400
                assert "SEC header" in resp.json()["detail"]

    def test_invalid_ticker_returns_422(self, client):
        resp = client.get("/api/company/INVALID!!!/filings")
        assert resp.status_code == 422


# ── Partial data ───────────────────────────────────────────────────────────


class TestFilingsPartialData:
    def test_no_tenk_returns_without_tenk_section(self, client, mock_filings_deps):
        """Companies without a 10-K (e.g., new IPOs) should still return."""
        mock_filings_deps["sec"].get_tenk_filing.side_effect = ValueError("No 10-K")
        data = client.get(
            "/api/company/NEWIPO/filings",
            headers={"X-Google-Api-Key": "test-key", "X-Sec-Header": "test@test.com"},
        ).json()
        assert "tenk" not in data
        # 10-Q and earnings should still be present
        assert "tenq" in data
