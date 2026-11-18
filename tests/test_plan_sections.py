"""_plan_sections — which filing sections get resolved (cache lookup / LLM call).

Pure logic over plain dicts shaped like ``_fetch_filing_data`` output.
"""

from api.routes.company import _plan_sections

TENK_META = {"accession": "0000320193-25-000001", "form_type": "10-K"}
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


def _cache_only(plan):
    """(form, atype) pairs planned cache-only: served from cache, never generated."""
    return {(p["form"], p["atype"]) for p in plan if p["cache_only"]}


class TestFailedExtractionIsCacheOnly:
    """A fetched-but-failed section stays planned so a cached analysis is still
    served, but is marked cache_only so a cache miss never reaches the LLM."""

    def test_all_extracted_plans_core_sections(self):
        plan = _plan_sections(_fetched())
        assert _atypes(plan) == [
            ("10-K", "risk_10k"), ("10-K", "mda_10k"), ("10-K", "balance"),
            ("10-Q", "risk_10q"), ("10-Q", "mda_10q"),
        ]
        assert _cache_only(plan) == set()

    def test_failed_tenk_risk_is_cache_only(self):
        assert _cache_only(_plan_sections(_fetched(risk=FAILED))) == {("10-K", "risk_10k")}

    def test_failed_tenk_mda_is_cache_only(self):
        assert _cache_only(_plan_sections(_fetched(mda=FAILED))) == {("10-K", "mda_10k")}

    def test_failed_tenq_sections_are_cache_only(self):
        assert _cache_only(_plan_sections(_fetched(tenq_risk=FAILED, tenq_mda=FAILED))) == {
            ("10-Q", "risk_10q"), ("10-Q", "mda_10q"),
        }

    def test_empty_balance_sheet_is_cache_only(self):
        # 20-F filers have no XBRL balance sheet → `balance_sheet_raw: {}`.
        assert _cache_only(_plan_sections(_fetched(balance={}))) == {("10-K", "balance")}

    def test_unparseable_earnings_is_cache_only(self):
        eightk = {
            "kind": "earnings",
            "raw": {"has_earnings": False, "reason": "lacks parseable EX-99.1"},
            "metadata": {"accession": "8k-acc"},
        }
        assert _cache_only(_plan_sections(_fetched(eightk=eightk))) == {("8-K", "earnings")}

    def test_parsed_earnings_generates(self):
        eightk = {
            "kind": "earnings",
            "raw": {"has_earnings": True, "context": "Q1"},
            "metadata": {"accession": "8k-acc"},
        }
        plan = _plan_sections(_fetched(eightk=eightk))
        assert ("8-K", "earnings") in _atypes(plan)
        assert _cache_only(plan) == set()

    def test_event_without_item_text_is_cache_only(self):
        # edgartools' to_context() always returns metadata, so context alone is
        # not real input — only the item text is.
        eightk = {
            "kind": "event",
            "raw": {"items": ["5.02"], "context": "8K: X Corp Current Report", "text": ""},
            "metadata": {"accession": "8k-acc"},
        }
        assert _cache_only(_plan_sections(_fetched(eightk=eightk))) == {("8-K", "event")}

    def test_event_with_item_text_generates(self):
        eightk = {
            "kind": "event",
            "raw": {"items": ["5.02"], "context": "ctx", "text": "CEO resigned."},
            "metadata": {"accession": "8k-acc"},
        }
        assert _cache_only(_plan_sections(_fetched(eightk=eightk))) == set()


class TestKeylessStillPlansCoreSections:
    """Keyless callers skip raw extraction but must still look up cached analyses."""

    CONDITIONAL = [
        ("10-K", "business"), ("10-K", "cybersecurity"), ("10-K", "legal"),
        ("10-K", "market_risk"), ("10-K", "income_stmt"), ("10-K", "cashflow"),
    ]

    def test_keyless_plans_core_and_8k(self):
        plan = _plan_sections(_keyless())
        assert [a for a in _atypes(plan) if a not in self.CONDITIONAL] == [
            ("10-K", "risk_10k"), ("10-K", "mda_10k"), ("10-K", "balance"),
            ("10-Q", "risk_10q"), ("10-Q", "mda_10q"), ("8-K", "earnings"),
        ]

    def test_keyless_core_is_never_cache_only(self):
        # Keyless input is unfetched, not failed — a miss must surface needs_key.
        plan = _plan_sections(_keyless())
        assert _cache_only([p for p in plan if (p["form"], p["atype"]) not in self.CONDITIONAL]) == set()

    def test_keyless_plans_conditional_10k_sections_cache_only(self):
        # The filing may not contain them, so a miss is silent (no needs_key).
        plan = _plan_sections(_keyless())
        assert [a for a in _atypes(plan) if a in self.CONDITIONAL] == self.CONDITIONAL
        assert _cache_only(plan) == set(self.CONDITIONAL)
        assert all(p["accession"] == TENK_META["accession"] for p in plan if p["form"] == "10-K")

    def test_keyless_foreign_filer_plans_no_conditional_sections(self):
        data = _keyless()
        data["tenk"] = {"metadata": {"accession": "20f-acc", "form_type": "20-F"}}
        assert not set(_atypes(_plan_sections(data))) & set(self.CONDITIONAL)

    def test_fetched_but_missing_sections_are_not_planned(self):
        # Keyed fetch found no Item 1 etc.: nothing to look up, nothing to generate.
        assert not set(_atypes(_plan_sections(_fetched()))) & set(self.CONDITIONAL)

    def test_keyless_event_is_never_cache_only(self):
        data = _keyless()
        data["eightk"] = {"kind": "event", "raw": {}, "metadata": {"accession": "8k-acc"}}
        assert ("8-K", "event") not in _cache_only(_plan_sections(data))
