"""`SECDataRetrieval.get_8k_item` handles blank and missing items without
asking edgartools for an empty key (FutureWarning today, SectionNotFoundError
in edgartools 6.0)."""

import warnings

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from tests.sec_fakes import FakeEightK


class _StrictEightK(FakeEightK):
    """Behaves like edgartools 6.0: a missing item raises a KeyError subclass."""

    def __getitem__(self, item: str) -> str | None:
        if item not in self._items:
            raise KeyError(item)
        return self._items[item]


class _CountingEightK(FakeEightK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookups: list[str] = []

    def __getitem__(self, item: str) -> str | None:
        self.lookups.append(item)
        return super().__getitem__(item)


def _retriever(eightk: FakeEightK) -> SECDataRetrieval:
    r = object.__new__(SECDataRetrieval)  # skip the EDGAR network lookup
    r.ticker = "AAPL"
    r._eightk_metadata = None
    r.get_eightk = lambda: eightk
    return r


def test_blank_item_never_reaches_edgartools():
    eightk = _CountingEightK({"5.02": "Director resigned."}, has_earnings=False, filed="2026-11-01")
    for blank in ("", "   ", None):
        result = _retriever(eightk).get_8k_item(blank)
        assert result["found"] is False
    assert eightk.lookups == []


def test_present_item_is_returned():
    eightk = FakeEightK({"5.02": "Director resigned."}, has_earnings=False, filed="2026-11-01")
    result = _retriever(eightk).get_8k_item(" 5.02 ")
    assert result == {"text": "Director resigned.", "metadata": {}, "found": True}


def test_missing_item_raising_keyerror_is_not_found_not_an_error():
    eightk = _StrictEightK({"5.02": "Director resigned."}, has_earnings=False, filed="2026-11-01")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _retriever(eightk).get_8k_item("2.02")
    assert result["found"] is False
    assert result["text"] == "Item 2.02 not found in 8-K"
