"""_plan_sections — which filing sections get resolved (cache lookup / LLM call).

Pure logic over plain dicts shaped like ``_fetch_filing_data`` output.
"""

from api.routes.company import _plan_sections

TENK_META = {"accession": "0000320193-25-000001"}
TENQ_META = {"accession": "0000320193-25-000002"}

FOUND = {"text": "Real section text", "found": True}
FAILED = {"text": "Error extracting 1A from 10-K: boom", "metadata": {}, "found": False}


def _atypes(plan):
    return [(p["form"], p["atype"]) for p in plan]


def _fetched(risk=FOUND, mda=FOUND, balance=None, tenq_risk=FOUND, tenq_mda=FOUND, eightk=None):
    return {
        "ticker": "AAPL",
        "tenk": {
            "metadata": TENK_META,
            "risk_raw": risk,
            "mda_raw": mda,
            "balance_sheet_raw": {"tenk": "Balance sheet"} if balance is None else balance,
        },
        "tenq": {"metadata": TENQ_META, "risk_raw": tenq_risk, "mda_raw": tenq_mda},
        "eightk": eightk or {"kind": "none", "reason": "No 8-K"},
    }


def _keyless():
    """fetch_raw=False shape: metadata only, no raw keys."""
    return {
        "ticker": "AAPL",
        "tenk": {"metadata": TENK_META},
        "tenq": {"metadata": TENQ_META},
        "eightk": {"kind": "earnings", "raw": {}, "metadata": {"accession": "8k-acc"}},
    }


class TestFailedExtractionSkipped:
    def test_all_extracted_plans_core_sections(self):
        assert _atypes(_plan_sections(_fetched())) == [
            ("10-K", "risk_10k"), ("10-K", "mda_10k"), ("10-K", "balance"),
            ("10-Q", "risk_10q"), ("10-Q", "mda_10q"),
        ]

    def test_failed_tenk_risk_not_planned(self):
        atypes = _atypes(_plan_sections(_fetched(risk=FAILED)))
        assert ("10-K", "risk_10k") not in atypes
        assert ("10-K", "mda_10k") in atypes

    def test_failed_tenk_mda_not_planned(self):
        assert ("10-K", "mda_10k") not in _atypes(_plan_sections(_fetched(mda=FAILED)))

    def test_failed_tenq_sections_not_planned(self):
        atypes = _atypes(_plan_sections(_fetched(tenq_risk=FAILED, tenq_mda=FAILED)))
        assert ("10-Q", "risk_10q") not in atypes
        assert ("10-Q", "mda_10q") not in atypes

    def test_empty_balance_sheet_not_planned(self):
        # 20-F filers have no XBRL balance sheet → `balance_sheet_raw: {}`.
        assert ("10-K", "balance") not in _atypes(_plan_sections(_fetched(balance={})))

    def test_unparseable_earnings_not_planned(self):
        eightk = {
            "kind": "earnings",
            "raw": {"has_earnings": False, "reason": "lacks parseable EX-99.1"},
            "metadata": {"accession": "8k-acc"},
        }
        assert ("8-K", "earnings") not in _atypes(_plan_sections(_fetched(eightk=eightk)))

    def test_parsed_earnings_planned(self):
        eightk = {
            "kind": "earnings",
            "raw": {"has_earnings": True, "context": "Q1"},
            "metadata": {"accession": "8k-acc"},
        }
        assert ("8-K", "earnings") in _atypes(_plan_sections(_fetched(eightk=eightk)))


class TestKeylessStillPlansCoreSections:
    """Keyless callers skip raw extraction but must still look up cached analyses."""

    def test_keyless_plans_core_and_8k(self):
        assert _atypes(_plan_sections(_keyless())) == [
            ("10-K", "risk_10k"), ("10-K", "mda_10k"), ("10-K", "balance"),
            ("10-Q", "risk_10q"), ("10-Q", "mda_10q"), ("8-K", "earnings"),
        ]
