"""Tests for 8-K routing in `_fetch_filing_data` and `_run_llm_analysis`.

Covers the three branches added to fix the dashboard 8-K visibility gap
(issue #31). Uses small dataclass-shaped fakes for the SEC retriever
rather than `MagicMock` so tests document the contract explicitly.
"""

from typing import Any
from unittest.mock import patch

import pytest

from api.routes import company as company_module


_TICKER = "AAPL"
_SEC_HEADER = "test@test.com"

_EIGHTK_META = {
    "form": "8-K",
    "cik": "320193",
    "accession": "0000320193-26-000789",
    "filing_date": "2026-02-15",
    "period_of_report": "2026-02-15",
    "company_name": "APPLE INC",
}


class _FakeMeta:
    """Stand-in for the FilingMetadata dataclass.

    Mirrors only the surface `_fetch_filing_data` reads: `to_dict()`, `cik`,
    `accession`. Using a hand-rolled fake keeps the test contract obvious
    instead of hiding behind `MagicMock`.
    """

    def __init__(self, payload: dict[str, str]):
        self._payload = payload
        self.cik = payload["cik"]
        self.accession = payload["accession"]

    def to_dict(self) -> dict[str, str]:
        return dict(self._payload)


class _FakeRetriever:
    """Configurable fake SEC retriever for `_fetch_filing_data` tests.

    Only methods touched by the 8-K branch are implemented; the 10-K and
    10-Q methods raise so those paths are skipped without polluting the
    `eightk` slot under test.
    """

    def __init__(
        self,
        overview: dict[str, Any],
        earnings: dict[str, Any] | None = None,
        item_text: str = "",
        eightk_meta: _FakeMeta | None = None,
    ):
        self._overview = overview
        self._earnings = earnings or {}
        self._item_text = item_text
        self._eightk_metadata = eightk_meta

    def get_tenk_filing(self):
        raise ValueError("no tenk in this fixture")

    def get_twentyf_filing(self):
        raise ValueError("no 20-F in this fixture")

    def get_tenq_filing(self):
        raise ValueError("no tenq in this fixture")

    def get_8k_overview(self) -> dict[str, Any]:
        return self._overview

    def get_8k_item(self, item: str) -> dict[str, Any]:
        return {"text": self._item_text, "metadata": _EIGHTK_META, "found": bool(self._item_text)}

    def get_earnings_data(self) -> dict[str, Any]:
        return self._earnings


def _run_fetch(retriever: _FakeRetriever) -> dict[str, Any]:
    """Run `_fetch_filing_data` with a patched retriever class."""
    with patch(
        "agents.sec_workflow.get_SEC_data.SECDataRetrieval",
        return_value=retriever,
    ):
        return company_module._fetch_filing_data(_TICKER, _SEC_HEADER)


# ── Branch 1: earnings 8-K ─────────────────────────────────────────────────


def test_fetch_filing_data_routes_earnings_eightk():
    meta = _FakeMeta(_EIGHTK_META)
    retriever = _FakeRetriever(
        overview={
            "found": True,
            "has_earnings": True,
            "items": ["2.02"],
            "content_type": "earnings",
            "is_amendment": False,
            "has_press_release": True,
            "date_of_report": "2026-01-30",
            "context": "Q1 2026 earnings",
            "metadata": _EIGHTK_META,
        },
        earnings={
            "has_earnings": True,
            "context": "Q1 2026 earnings",
            "detected_scale": "millions",
            "metadata": _EIGHTK_META,
        },
        eightk_meta=meta,
    )

    result = _run_fetch(retriever)
    eightk = result["eightk"]

    assert eightk["kind"] == "earnings"
    assert eightk["raw"]["has_earnings"] is True
    assert eightk["metadata"]["accession"] == _EIGHTK_META["accession"]
    assert eightk["metadata"]["form_type"] == "8-K"
    assert "edgar_url" in eightk["metadata"]


# ── Branch 2: material event 8-K ───────────────────────────────────────────


def test_fetch_filing_data_routes_material_event_eightk():
    meta = _FakeMeta(_EIGHTK_META)
    retriever = _FakeRetriever(
        overview={
            "found": True,
            "has_earnings": False,
            "items": ["5.02"],
            "content_type": "director_change",
            "is_amendment": False,
            "has_press_release": False,
            "date_of_report": "2026-02-15",
            "context": "Departure of CEO; appointment of interim CEO.",
            "metadata": _EIGHTK_META,
        },
        item_text="On February 15, 2026, the CEO notified the board of his resignation...",
        eightk_meta=meta,
    )

    result = _run_fetch(retriever)
    eightk = result["eightk"]

    assert eightk["kind"] == "event"
    raw = eightk["raw"]
    # Shape must match what `_tool_material_event_summary` builds so
    # `analyze_material_event` accepts it without adaptation.
    assert raw["content_type"] == "director_change"
    assert raw["items"] == ["5.02"]
    assert raw["text"].startswith("On February 15")
    assert raw["context"]
    assert raw["metadata"]["accession"] == _EIGHTK_META["accession"]


# ── Branch 3: no 8-K found ─────────────────────────────────────────────────


def test_fetch_filing_data_eightk_none_when_overview_not_found():
    retriever = _FakeRetriever(
        overview={"found": False, "text": "No 8-K available", "metadata": {}},
    )

    result = _run_fetch(retriever)
    eightk = result["eightk"]

    assert eightk["kind"] == "none"
    assert "reason" in eightk
    assert "raw" not in eightk


def test_fetch_filing_data_eightk_none_when_overview_raises():
    """Exception during overview fetch should land in the `none` branch with a reason."""

    class _Boom(_FakeRetriever):
        def get_8k_overview(self):
            raise RuntimeError("edgartools blew up")

    retriever = _Boom(overview={})
    result = _run_fetch(retriever)
    eightk = result["eightk"]

    assert eightk["kind"] == "none"
    assert "edgartools blew up" in eightk["reason"]


# ── Dispatch: `_run_llm_analysis` event branch ─────────────────────────────


def test_run_llm_analysis_dispatches_event_to_material_event_analyzer():
    """analysis_type='event' must call `processor.analyze_material_event`."""
    raw = {
        "content_type": "agreement",
        "items": ["1.01"],
        "context": "Material definitive agreement signed.",
        "text": "Item 1.01 — material agreement...",
        "metadata": _EIGHTK_META,
    }

    class _StubResult:
        def model_dump(self):
            return {"event_type": "Material Agreement", "summary": "ok"}

    class _StubProcessor:
        def __init__(self, llm):
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def analyze_material_event(self, ticker: str, data: dict[str, Any]):
            self.calls.append((ticker, data))
            return _StubResult()

    with (
        patch("agents.llm_factory.create_llm", return_value=object()),
        patch(
            "agents.sec_workflow.sec_llm_models.SECDocumentProcessor",
            _StubProcessor,
        ),
    ):
        result = company_module._run_llm_analysis(
            _TICKER, "event", raw, "gemini-2.5-flash", "fake-key",
        )

    assert result == {"event_type": "Material Agreement", "summary": "ok"}


def test_run_llm_analysis_unknown_type_still_raises():
    with (
        patch("agents.llm_factory.create_llm", return_value=object()),
        patch("agents.sec_workflow.sec_llm_models.SECDocumentProcessor"),
    ):
        with pytest.raises(ValueError, match="Unknown analysis type"):
            company_module._run_llm_analysis(
                _TICKER, "not_a_real_type", {}, "gemini-2.5-flash", "fake-key",
            )
