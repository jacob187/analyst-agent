"""Keyless visitors see the conditional 10-K analyses a keyed caller already cached.

Business / cybersecurity / legal / market-risk / income-statement / cash-flow
are planned only from raw text, which a keyless fetch skips. They are planned
as cache-only lookups instead: served on a hit, silently omitted on a miss
(the filing may not contain them) — never a ``needs_key`` placeholder.

Real temp SQLite for the analysis cache; SEC retrieval and the LLM are faked at
their boundaries.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import byok_header, model_where
from tests.sec_fakes import CountingChatModel
from tests.test_filings_endpoint import (
    _MOCK_8K_METADATA,
    _MOCK_TENK_METADATA,
    _MOCK_TENQ_METADATA,
    _STATEMENT,
    _found,
    _no_keys,
    _stream_events,
)

SIX = ("business", "cybersecurity", "legal", "market_risk", "income_stmt", "cashflow")
SIX_STEPS = {f"10-K/{a}" for a in SIX}
_BASE = "/api/company/AAPL/filings"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db(tmp_path, monkeypatch):
    import api.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "anon_cache.db")
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.run(db_module.init_db())
    yield
    asyncio.run(db_module.close_db())


@pytest.fixture
def edgar(db):
    """A filer whose 10-K has every conditional section; yields the retriever
    mock and the fake chat model."""
    tenk_meta, tenq_meta = MagicMock(), MagicMock()
    tenk_meta.to_dict.return_value = _MOCK_TENK_METADATA
    tenk_meta.cik, tenk_meta.accession = "320193", _MOCK_TENK_METADATA["accession"]
    tenq_meta.to_dict.return_value = _MOCK_TENQ_METADATA
    tenq_meta.cik, tenq_meta.accession = "320193", _MOCK_TENQ_METADATA["accession"]

    llm = CountingChatModel()
    with (
        patch("agents.sec_workflow.get_SEC_data.SECDataRetrieval") as MockSEC,
        patch("agents.llm_factory.create_llm", return_value=llm),
    ):
        sec = MagicMock()
        sec._tenk_metadata, sec._tenq_metadata = tenk_meta, tenq_meta
        meta = {"10-K": _MOCK_TENK_METADATA, "10-Q": _MOCK_TENQ_METADATA}
        sec.get_risk_factors_raw.side_effect = lambda form="10-K": {
            "text": "Risk text", "metadata": meta[form], "found": True,
        }
        sec.get_mda_raw.side_effect = lambda form="10-K": {
            "text": "MD&A text", "metadata": meta[form], "found": True,
        }
        sec.balance_sheet_text.return_value = ("Balance sheet text", tenk_meta)
        sec.get_8k_overview.return_value = {"found": False, "text": "No 8-K"}
        sec.get_business_raw.return_value = _found("Item 1 text")
        sec.get_cybersecurity_raw.return_value = _found("Item 1C text")
        sec.get_legal_proceedings_raw.return_value = _found("Item 3 text")
        sec.get_market_risk_raw.return_value = _found("Item 7A text")
        sec.get_income_statement.return_value = _STATEMENT
        sec.get_cashflow_statement.return_value = _STATEMENT
        MockSEC.return_value = sec
        yield sec, llm


def _keyed():
    return byok_header(model_where().provider)


def _tenk(resp):
    assert resp.status_code == 200
    return resp.json()["tenk"]


class TestAnonServedCachedSections:
    def test_json_serves_what_a_keyed_request_cached(self, client, edgar):
        sec, llm = edgar
        keyed = _tenk(client.get(_BASE, headers=_keyed()))
        assert set(SIX) <= set(keyed)
        calls = llm.calls

        sec.get_business_raw.reset_mock()
        with _no_keys():
            anon = _tenk(client.get(_BASE))

        for atype in SIX:
            assert anon[atype] == keyed[atype]
            assert anon[atype]["filing_reference"]["accession"] == _MOCK_TENK_METADATA["accession"]
        assert llm.calls == calls  # served from cache, nothing generated
        sec.get_business_raw.assert_not_called()  # still no raw extraction

    def test_stream_serves_what_a_keyed_request_cached(self, client, edgar):
        _sec, llm = edgar
        keyed = {
            e["key"]: e["data"]
            for e in _stream_events(client, _BASE + "/stream", _keyed())
            if e.get("type") == "section"
        }
        assert set(SIX) <= set(keyed)
        calls = llm.calls

        with _no_keys():
            events = _stream_events(client, _BASE + "/stream")

        served = {e["key"]: e["data"] for e in events if e.get("type") == "section"}
        for atype in SIX:
            assert served[atype] == keyed[atype]
        cached_steps = {
            e["step"] for e in events if e.get("type") == "progress" and e["status"] == "cached"
        }
        assert SIX_STEPS <= cached_steps
        assert llm.calls == calls

    def test_sections_the_filing_lacks_stay_omitted(self, client, edgar):
        sec, _llm = edgar
        sec.get_legal_proceedings_raw.return_value = {"found": False}
        sec.get_cashflow_statement.return_value = None
        _tenk(client.get(_BASE, headers=_keyed()))

        with _no_keys():
            anon = _tenk(client.get(_BASE))

        assert {"business", "cybersecurity", "market_risk", "income_stmt"} <= set(anon)
        assert "legal" not in anon
        assert "cashflow" not in anon


class TestAnonNothingCached:
    def test_json_omits_the_six(self, client, edgar):
        _sec, llm = edgar
        with _no_keys():
            anon = _tenk(client.get(_BASE))
        assert not set(SIX) & set(anon)
        assert llm.calls == 0

    def test_stream_omits_the_six_and_emits_no_needs_key_for_them(self, client, edgar):
        _sec, llm = edgar
        with _no_keys():
            events = _stream_events(client, _BASE + "/stream")

        progress = [e for e in events if e.get("type") == "progress"]
        assert not SIX_STEPS & {e["step"] for e in progress}
        assert not set(SIX) & {e["key"] for e in events if e.get("type") == "section"}
        # The core sections still report needs_key.
        assert ("10-K/risk_10k", "needs_key") in {(e["step"], e["status"]) for e in progress}
        assert llm.calls == 0
