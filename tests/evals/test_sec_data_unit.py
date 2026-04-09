"""Unit tests for SEC data retrieval fixes — no API keys required.

Marker: eval_unit

Tests:
1. _is_substantive() edge cases
2. _fetch_best_section() fallback logic
3. get_section() uses correct item keys for each form
4. New retrieval methods delegate to correct item codes
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.tools.sec_tools import _is_substantive, _fetch_best_section
from agents.sec_workflow.get_SEC_data import SECDataRetrieval, _TENK_ITEM_KEYS, _TENQ_ITEM_KEYS
from edgar import CompanyNotFoundError


# ── _is_substantive ──────────────────────────────────────────────────────────

@pytest.mark.eval_unit
class TestIsSubstantive:
    """_is_substantive detects boilerplate 10-Q risk factor text."""

    def test_long_text_is_always_substantive(self):
        assert _is_substantive("A" * 600)

    def test_short_text_without_boilerplate_is_substantive(self):
        assert _is_substantive("Significant new regulatory risk from AI legislation.")

    def test_no_material_changes_boilerplate(self):
        assert not _is_substantive("There have been no material changes to the risk factors.")

    def test_not_materially_changed_boilerplate(self):
        assert not _is_substantive("Our risk profile has not materially changed since the last filing.")

    def test_incorporated_by_reference_boilerplate(self):
        assert not _is_substantive("Risk factors are incorporated by reference from our 10-K.")

    def test_long_text_with_boilerplate_phrase_is_still_substantive(self):
        # 500+ chars with boilerplate phrase embedded — still substantive by length rule.
        text = "There have been no material changes " + ("X" * 470)
        assert _is_substantive(text)

    def test_empty_string_is_not_substantive(self):
        assert not _is_substantive("")


# ── _fetch_best_section ──────────────────────────────────────────────────────

@pytest.mark.eval_unit
class TestFetchBestSection:
    """_fetch_best_section prefers 10-Q when substantive, falls back to 10-K."""

    def _make_result(self, found: bool, text: str) -> dict:
        return {"found": found, "text": text, "metadata": {}}

    def test_uses_10q_when_substantive(self):
        tenq_result = self._make_result(True, "Q" * 600)
        tenk_result = self._make_result(True, "K" * 600)
        fetch_fn = MagicMock(side_effect=lambda form: tenq_result if form == "10-Q" else tenk_result)

        result = _fetch_best_section(fetch_fn)

        assert result["text"] == "Q" * 600
        fetch_fn.assert_called_once_with("10-Q")

    def test_falls_back_to_10k_when_10q_boilerplate(self):
        tenq_result = self._make_result(True, "No material changes.")
        tenk_result = self._make_result(True, "K" * 600)
        fetch_fn = MagicMock(side_effect=lambda form: tenq_result if form == "10-Q" else tenk_result)

        result = _fetch_best_section(fetch_fn)

        assert result["text"] == "K" * 600

    def test_falls_back_to_10k_when_10q_not_found(self):
        tenq_result = self._make_result(False, "")
        tenk_result = self._make_result(True, "K" * 600)
        fetch_fn = MagicMock(side_effect=lambda form: tenq_result if form == "10-Q" else tenk_result)

        result = _fetch_best_section(fetch_fn)

        assert result["text"] == "K" * 600


# ── Item key mappings ────────────────────────────────────────────────────────

@pytest.mark.eval_unit
class TestItemKeyMappings:
    """_TENK_ITEM_KEYS and _TENQ_ITEM_KEYS contain correct edgartools keys."""

    def test_tenk_risk_factors_key(self):
        assert _TENK_ITEM_KEYS["1A"] == "Item 1A"

    def test_tenk_mda_key(self):
        assert _TENK_ITEM_KEYS["7"] == "Item 7"

    def test_tenk_business_key(self):
        assert _TENK_ITEM_KEYS["1"] == "Item 1"

    def test_tenk_cybersecurity_key(self):
        assert _TENK_ITEM_KEYS["1C"] == "Item 1C"

    def test_tenk_legal_proceedings_key(self):
        assert _TENK_ITEM_KEYS["3"] == "Item 3"

    def test_tenk_market_risk_key(self):
        assert _TENK_ITEM_KEYS["7A"] == "Item 7A"

    def test_tenq_risk_factors_key(self):
        assert _TENQ_ITEM_KEYS["1A"] == "Item 1A"

    def test_tenq_mda_key(self):
        assert _TENQ_ITEM_KEYS["2"] == "Item 2"

    def test_tenq_market_risk_key(self):
        assert _TENQ_ITEM_KEYS["3"] == "Item 3"


# ── get_section() uses correct item keys ─────────────────────────────────────

@pytest.mark.eval_unit
class TestGetSection:
    """get_section() calls filing_obj[key] with the correct item key."""

    def _make_retriever(self, tenk_text: str = "text", tenq_text: str = "text") -> SECDataRetrieval:
        retriever = MagicMock(spec=SECDataRetrieval)
        retriever.ticker = "TEST"

        mock_tenk = MagicMock()
        mock_tenk.__getitem__ = MagicMock(return_value=tenk_text)
        retriever.get_tenk.return_value = mock_tenk

        mock_tenq = MagicMock()
        mock_tenq.__getitem__ = MagicMock(return_value=tenq_text)
        retriever.get_tenq.return_value = mock_tenq

        from agents.sec_workflow.get_SEC_data import FilingMetadata
        meta = FilingMetadata("10-K", "123", "acc-001", "2025-01-01", "2024-12-31", "Test Corp")
        retriever._tenk_metadata = meta
        retriever._tenq_metadata = FilingMetadata("10-Q", "123", "acc-002", "2025-03-01", "2025-03-31", "Test Corp")

        # Delegate get_section to the real implementation
        retriever.get_section = lambda form, item: SECDataRetrieval.get_section(retriever, form, item)
        return retriever

    def test_10k_risk_factors_uses_item_1a(self):
        r = self._make_retriever()
        r.get_section("10-K", "1A")
        r.get_tenk.return_value.__getitem__.assert_called_with("Item 1A")

    def test_10k_mda_uses_item_7(self):
        r = self._make_retriever()
        r.get_section("10-K", "7")
        r.get_tenk.return_value.__getitem__.assert_called_with("Item 7")

    def test_10q_risk_factors_uses_item_1a(self):
        r = self._make_retriever()
        r.get_section("10-Q", "1A")
        r.get_tenq.return_value.__getitem__.assert_called_with("Item 1A")

    def test_10q_mda_uses_item_2(self):
        r = self._make_retriever()
        r.get_section("10-Q", "2")
        r.get_tenq.return_value.__getitem__.assert_called_with("Item 2")

    def test_unsupported_item_returns_not_found(self):
        r = self._make_retriever()
        result = r.get_section("10-Q", "99")
        assert not result["found"]
        assert "not supported" in result["text"]

    def test_none_text_returns_not_found(self):
        r = self._make_retriever(tenq_text=None)
        result = r.get_section("10-Q", "1A")
        assert not result["found"]


# ── New retrieval methods route to correct item codes ────────────────────────

@pytest.mark.eval_unit
class TestNewRetrievalMethods:
    """get_business_raw, get_cybersecurity_raw, etc. delegate to get_section correctly."""

    def _make_retriever_with_real_methods(self):
        r = MagicMock(spec=SECDataRetrieval)
        r.get_section = MagicMock(return_value={"text": "x", "metadata": {}, "found": True})

        # Bind real method implementations
        r.get_business_raw = lambda: SECDataRetrieval.get_business_raw(r)
        r.get_cybersecurity_raw = lambda: SECDataRetrieval.get_cybersecurity_raw(r)
        r.get_legal_proceedings_raw = lambda: SECDataRetrieval.get_legal_proceedings_raw(r)
        r.get_market_risk_raw = lambda form="10-K": SECDataRetrieval.get_market_risk_raw(r, form)
        r.get_mda_raw = lambda form="10-K": SECDataRetrieval.get_mda_raw(r, form)
        r.get_risk_factors_raw = lambda form="10-K": SECDataRetrieval.get_risk_factors_raw(r, form)
        return r

    def test_get_business_raw_uses_tenk_item_1(self):
        r = self._make_retriever_with_real_methods()
        r.get_business_raw()
        r.get_section.assert_called_with("10-K", "1")

    def test_get_cybersecurity_raw_uses_tenk_item_1c(self):
        r = self._make_retriever_with_real_methods()
        r.get_cybersecurity_raw()
        r.get_section.assert_called_with("10-K", "1C")

    def test_get_legal_proceedings_raw_uses_tenk_item_3(self):
        r = self._make_retriever_with_real_methods()
        r.get_legal_proceedings_raw()
        r.get_section.assert_called_with("10-K", "3")

    def test_get_market_risk_raw_tenk_uses_7a(self):
        r = self._make_retriever_with_real_methods()
        r.get_market_risk_raw("10-K")
        r.get_section.assert_called_with("10-K", "7A")

    def test_get_market_risk_raw_tenq_uses_item_3(self):
        r = self._make_retriever_with_real_methods()
        r.get_market_risk_raw("10-Q")
        r.get_section.assert_called_with("10-Q", "3")

    def test_get_mda_raw_tenk_uses_item_7(self):
        r = self._make_retriever_with_real_methods()
        r.get_mda_raw("10-K")
        r.get_section.assert_called_with("10-K", "7")

    def test_get_mda_raw_tenq_uses_item_2(self):
        r = self._make_retriever_with_real_methods()
        r.get_mda_raw("10-Q")
        r.get_section.assert_called_with("10-Q", "2")


# ── CompanyNotFoundError handling ─────────────────────────────────────────────

@pytest.mark.eval_unit
class TestCompanyNotFoundError:
    """SECDataRetrieval translates CompanyNotFoundError to ValueError."""

    def test_invalid_ticker_raises_value_error(self):
        with patch("agents.sec_workflow.get_SEC_data.Company") as MockCompany:
            MockCompany.side_effect = CompanyNotFoundError("FAKE", suggestions=[])
            with pytest.raises(ValueError, match="Company not found"):
                SECDataRetrieval("FAKE", "test test@test.com")

    def test_invalid_ticker_includes_suggestions(self):
        suggestions = [{"ticker": "AAPL", "company": "Apple Inc"}]
        with patch("agents.sec_workflow.get_SEC_data.Company") as MockCompany:
            MockCompany.side_effect = CompanyNotFoundError(
                "AAPL1", suggestions=suggestions
            )
            with pytest.raises(ValueError, match="Did you mean: AAPL"):
                SECDataRetrieval("AAPL1", "test test@test.com")

    def test_valid_ticker_no_error(self):
        with patch("agents.sec_workflow.get_SEC_data.Company") as MockCompany:
            with patch("agents.sec_workflow.get_SEC_data.set_identity"):
                MockCompany.return_value = MagicMock()
                retriever = SECDataRetrieval("AAPL", "test test@test.com")
                assert retriever.ticker == "AAPL"


# ── 8-K retrieval methods ────────────────────────────────────────────────────

@pytest.mark.eval_unit
class TestGet8kOverview:
    """get_8k_overview returns expected dict shape from EightK object."""

    def _make_retriever_with_eightk(self, items=None, content_type="earnings",
                                     has_earnings=True):
        r = MagicMock(spec=SECDataRetrieval)
        mock_eightk = MagicMock()
        mock_eightk.items = items or ["Item 2.02", "Item 9.01"]
        mock_eightk.content_type = content_type
        mock_eightk.is_amendment = False
        mock_eightk.has_earnings = has_earnings
        mock_eightk.has_press_release = has_earnings
        mock_eightk.date_of_report = "April 01, 2026"
        mock_eightk.to_context.return_value = "8-K context summary"
        r.get_eightk.return_value = mock_eightk
        r._eightk_metadata = MagicMock()
        r._eightk_metadata.to_dict.return_value = {"form": "8-K", "filing_date": "2026-04-01"}
        r.get_8k_overview = lambda: SECDataRetrieval.get_8k_overview(r)
        return r

    def test_returns_expected_fields(self):
        r = self._make_retriever_with_eightk()
        result = r.get_8k_overview()
        assert result["found"] is True
        assert result["content_type"] == "earnings"
        assert result["has_earnings"] is True
        assert result["items"] == ["Item 2.02", "Item 9.01"]
        assert result["context"] == "8-K context summary"

    def test_non_earnings_event(self):
        r = self._make_retriever_with_eightk(
            items=["Item 5.02", "Item 9.01"],
            content_type="director_change",
            has_earnings=False,
        )
        result = r.get_8k_overview()
        assert result["content_type"] == "director_change"
        assert result["has_earnings"] is False


@pytest.mark.eval_unit
class TestGet8kItem:
    """get_8k_item dispatches to EightK.__getitem__ correctly."""

    def _make_retriever(self, item_text="Earnings release text"):
        r = MagicMock(spec=SECDataRetrieval)
        mock_eightk = MagicMock()
        mock_eightk.__getitem__ = MagicMock(return_value=item_text)
        r.get_eightk.return_value = mock_eightk
        r._eightk_metadata = MagicMock()
        r._eightk_metadata.to_dict.return_value = {"form": "8-K"}
        r.get_8k_item = lambda item: SECDataRetrieval.get_8k_item(r, item)
        return r

    def test_found_item_returns_text(self):
        r = self._make_retriever("Item 2.02 content here")
        result = r.get_8k_item("2.02")
        assert result["found"] is True
        assert "Item 2.02 content here" in result["text"]

    def test_missing_item_returns_not_found(self):
        r = self._make_retriever(None)
        result = r.get_8k_item("1.01")
        assert result["found"] is False


@pytest.mark.eval_unit
class TestGetEarningsData:
    """get_earnings_data handles both earnings and non-earnings 8-Ks."""

    def _make_retriever(self, has_earnings=True):
        r = MagicMock(spec=SECDataRetrieval)
        mock_eightk = MagicMock()
        mock_eightk.has_earnings = has_earnings
        r._eightk_metadata = MagicMock()
        r._eightk_metadata.to_dict.return_value = {"form": "8-K"}

        if has_earnings:
            mock_earnings = MagicMock()
            mock_earnings.detected_scale = "MILLIONS"
            mock_earnings.to_context.return_value = "Earnings context"
            # income_statement returns a FinancialTable with .dataframe
            mock_income = MagicMock()
            mock_income.dataframe.to_json.return_value = '{"columns":[],"data":[]}'
            mock_earnings.income_statement = mock_income
            mock_earnings.balance_sheet = None
            mock_earnings.cash_flow_statement = None
            mock_eightk.earnings = mock_earnings

        r.get_eightk.return_value = mock_eightk
        r.get_earnings_data = lambda: SECDataRetrieval.get_earnings_data(r)
        return r

    def test_has_earnings_returns_data(self):
        r = self._make_retriever(has_earnings=True)
        result = r.get_earnings_data()
        assert result["has_earnings"] is True
        assert "income_statement" in result
        assert result["detected_scale"] == "MILLIONS"

    def test_no_earnings_returns_reason(self):
        r = self._make_retriever(has_earnings=False)
        result = r.get_earnings_data()
        assert result["has_earnings"] is False
        assert "reason" in result


@pytest.mark.eval_unit
class TestGetEightkFiling:
    """get_eightk_filing raises ValueError when no 8-K exists."""

    def test_no_8k_raises_value_error(self):
        r = MagicMock(spec=SECDataRetrieval)
        r.ticker = "FAKE"
        r._eightk_filing = None
        r._fetch_latest_eightk_filing = MagicMock(return_value=None)
        r.get_eightk_filing = lambda: SECDataRetrieval.get_eightk_filing(r)
        with pytest.raises(ValueError, match="No 8-K available"):
            r.get_eightk_filing()
