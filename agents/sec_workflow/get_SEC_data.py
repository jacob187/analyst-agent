from edgar import Company, CompanyNotFoundError, set_identity
import json
from typing import Literal, Optional, Dict, Any
from datetime import datetime

# Item code → edgartools __getitem__ key for each form.
# 10-K: items are unique across all parts; keys are "Item X" strings.
# 10-Q: installed edgartools 3.x resolves via chunked_document, so bare
#        "Item X" keys work.  "Item 3" = Market Risk (Part I),
#        "Item 1A" = Risk Factors (Part II).
_TENK_ITEM_KEYS: Dict[str, str] = {
    "1":  "Item 1",   # Business
    "1A": "Item 1A",  # Risk Factors
    "1C": "Item 1C",  # Cybersecurity (SEC-mandated since 2023)
    "3":  "Item 3",   # Legal Proceedings
    "7":  "Item 7",   # MD&A
    "7A": "Item 7A",  # Quantitative / Qualitative Market Risk
}

_TENQ_ITEM_KEYS: Dict[str, str] = {
    "1A": "Item 1A",  # Risk Factors (Part II)
    "2":  "Item 2",   # MD&A (Part I)
    "3":  "Item 3",   # Market Risk (Part I)
    "4":  "Item 4",   # Controls and Procedures (Part I)
    "5":  "Item 5",   # Other Information (Part II — insider trading etc.)
}


class FilingMetadata:
    """Metadata for SEC filings to track provenance."""

    def __init__(
        self,
        form: str,
        cik: str,
        accession: str,
        filing_date: str,
        period_of_report: str,
        company_name: str,
    ):
        self.form = form
        self.cik = cik
        self.accession = accession
        self.filing_date = filing_date
        self.period_of_report = period_of_report
        self.company_name = company_name

    def to_dict(self) -> Dict[str, str]:
        return {
            "form": self.form,
            "cik": self.cik,
            "accession": self.accession,
            "filing_date": self.filing_date,
            "period_of_report": self.period_of_report,
            "company_name": self.company_name,
        }

    def __str__(self) -> str:
        if self.form == "10-K":
            return f"Form {self.form} filed {self.filing_date}, Period: Year ended {self.period_of_report}"
        elif self.form == "10-Q":
            return f"Form {self.form} filed {self.filing_date}, Period: Quarter ended {self.period_of_report}"
        else:
            return f"Form {self.form} filed {self.filing_date}, Period: {self.period_of_report}"


class SECDataRetrieval:
    def __init__(self, ticker: str, sec_header: str):
        set_identity(sec_header)
        self.ticker = ticker
        try:
            self.company = Company(ticker)
        except CompanyNotFoundError as e:
            suggestions = [s["ticker"] for s in e.suggestions[:3]]
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Company not found: '{ticker}'.{hint}") from e
        # Lazy caches
        self._tenk_filing = None
        self._tenq_filing = None
        self._eightk_filing = None
        self._twentyf_filing = None
        self._tenk_obj = None
        self._tenq_obj = None
        self._eightk_obj = None
        self._twentyf_obj = None
        self._tenk_metadata = None
        self._tenq_metadata = None
        self._eightk_metadata = None
        self._twentyf_metadata = None

    def check_filing_availability(self) -> Dict[str, Any]:
        """Check what filings are available for this company.

        Returns:
            Dict with ticker, company_name, has_10k, and has_10q fields.
        """
        result = {
            "ticker": self.ticker,
            "company_name": self.company.name,
            "is_foreign": False,
            "has_10k": False,
            "has_10q": False,
            "has_8k": False,
            "has_20f": False,
        }
        try:
            result["is_foreign"] = self.company.is_foreign
        except Exception:
            pass
        try:
            filing_10k = self.company.latest(form="10-K")
            result["has_10k"] = filing_10k is not None
        except Exception:
            pass
        try:
            filing_10q = self.company.latest(form="10-Q")
            result["has_10q"] = filing_10q is not None
        except Exception:
            pass
        try:
            filing_8k = self.company.get_filings(form="8-K").latest(1)
            result["has_8k"] = filing_8k is not None
        except Exception:
            pass
        try:
            filing_20f = self.company.latest(form="20-F")
            result["has_20f"] = filing_20f is not None
        except Exception:
            pass
        return result

    # Lazy getters for filings (EntityFiling objects)
    def get_tenk_filing(self):
        if self._tenk_filing is None:
            self._tenk_filing = self._fetch_company_tenk_filing()
            if self._tenk_filing is None:
                raise ValueError(f"No 10-K available for {self.ticker}")
        return self._tenk_filing

    def get_tenq_filing(self):
        if self._tenq_filing is None:
            self._tenq_filing = self._fetch_company_tenq_filing()
            if self._tenq_filing is None:
                raise ValueError(f"No 10-Q available for {self.ticker}")
        return self._tenq_filing

    # Lazy getters for filing objects (TenK/TenQ objects)
    def get_tenk(self):
        if self._tenk_obj is None:
            filing = self.get_tenk_filing()
            self._tenk_obj = filing.obj()
        return self._tenk_obj

    def get_tenq(self):
        if self._tenq_obj is None:
            filing = self.get_tenq_filing()
            self._tenq_obj = filing.obj()
        return self._tenq_obj

    def get_eightk_filing(self):
        if self._eightk_filing is None:
            self._eightk_filing = self._fetch_latest_eightk_filing()
            if self._eightk_filing is None:
                raise ValueError(f"No 8-K available for {self.ticker}")
        return self._eightk_filing

    def get_eightk(self):
        if self._eightk_obj is None:
            filing = self.get_eightk_filing()
            self._eightk_obj = filing.obj()
        return self._eightk_obj

    def get_twentyf_filing(self):
        if self._twentyf_filing is None:
            self._twentyf_filing = self._fetch_company_twentyf_filing()
            if self._twentyf_filing is None:
                raise ValueError(f"No 20-F available for {self.ticker}")
        return self._twentyf_filing

    def get_twentyf(self):
        """Return parsed TwentyF object.

        edgartools v5 TwentyF provides named accessors: .risk_factors,
        .management_discussion, .balance_sheet, .income_statement,
        .business, .key_information, .financial_information.
        """
        if self._twentyf_obj is None:
            filing = self.get_twentyf_filing()
            self._twentyf_obj = filing.obj()
        return self._twentyf_obj

    # Private fetchers
    def _fetch_company_tenk_filing(self):
        filing = self.company.latest(form="10-K")
        if filing is None:
            print(f"No 10-K filing found for {self.company.name}")
            return None
        print("10-K filing found")

        # Extract metadata from the filing (EntityFiling object)
        self._tenk_metadata = FilingMetadata(
            form=filing.form,
            cik=str(filing.cik),
            accession=filing.accession_number,
            filing_date=str(filing.filing_date),
            period_of_report=str(filing.period_of_report),
            company_name=filing.company,
        )
        return filing

    def _fetch_company_tenq_filing(self):
        filing = self.company.latest(form="10-Q")
        if filing is None:
            print(f"No 10-Q filing found for {self.company.name}")
            return None
        print("10-Q filing found")

        # Extract metadata from the filing (EntityFiling object)
        self._tenq_metadata = FilingMetadata(
            form=filing.form,
            cik=str(filing.cik),
            accession=filing.accession_number,
            filing_date=str(filing.filing_date),
            period_of_report=str(filing.period_of_report),
            company_name=filing.company,
        )
        return filing

    def _fetch_company_twentyf_filing(self):
        filing = self.company.latest(form="20-F")
        if filing is None:
            print(f"No 20-F filing found for {self.company.name}")
            return None
        print("20-F filing found")

        self._twentyf_metadata = FilingMetadata(
            form=filing.form,
            cik=str(filing.cik),
            accession=filing.accession_number,
            filing_date=str(filing.filing_date),
            period_of_report=str(filing.period_of_report),
            company_name=filing.company,
        )
        return filing

    def _fetch_latest_eightk_filing(self):
        filing = self.company.get_filings(form="8-K").latest(1)
        if filing is None:
            print(f"No 8-K filing found for {self.company.name}")
            return None
        print("8-K filing found")

        self._eightk_metadata = FilingMetadata(
            form=filing.form,
            cik=str(filing.cik),
            accession=filing.accession_number,
            filing_date=str(filing.filing_date),
            period_of_report=str(filing.period_of_report),
            company_name=filing.company,
        )
        return filing

    # ── 8-K public methods ────────────────────────────────────────────────

    def get_8k_overview(self) -> Dict[str, Any]:
        """Overview of the latest 8-K: items, content type, earnings flag, dates.

        The ``content_type`` field is rule-based (no LLM call) and classifies the
        filing as 'earnings', 'cybersecurity', 'director_change', etc.
        ``to_context(detail="standard")`` returns a ~300-token LLM-friendly summary.
        """
        try:
            eightk = self.get_eightk()
            return {
                "items": eightk.items,
                "content_type": eightk.content_type,
                "is_amendment": eightk.is_amendment,
                "has_earnings": eightk.has_earnings,
                "has_press_release": eightk.has_press_release,
                "date_of_report": eightk.date_of_report,
                "context": eightk.to_context(detail="standard"),
                "metadata": self._eightk_metadata.to_dict() if self._eightk_metadata else {},
                "found": True,
            }
        except Exception as e:
            return {"found": False, "text": f"Error fetching 8-K overview: {e}", "metadata": {}}

    def get_8k_item(self, item: str) -> Dict[str, Any]:
        """Extract text for a specific 8-K item number (e.g. '2.02', '1.01').

        Uses ``EightK.__getitem__`` which accepts formats like '2.02',
        'Item 2.02', or 'item_202'.  Returns the same dict shape as
        ``get_section`` for consistency.
        """
        try:
            eightk = self.get_eightk()
            text = eightk[item]
            found = text is not None and bool(str(text).strip())
            return {
                "text": str(text) if found else f"Item {item} not found in 8-K",
                "metadata": self._eightk_metadata.to_dict() if self._eightk_metadata else {},
                "found": found,
            }
        except Exception as e:
            return {"text": f"Error extracting 8-K item {item}: {e}", "metadata": {}, "found": False}

    def get_earnings_data(self) -> Dict[str, Any]:
        """Structured earnings from 8-K Item 2.02 + EX-99.1 press release.

        Uses edgartools v5 ``EarningsRelease`` to parse income statement,
        balance sheet, and cash flow tables from the press release exhibit.
        Returns raw JSON-serializable dicts — the LLM analysis layer in
        ``sec_llm_models.py`` handles interpretation.
        """
        try:
            eightk = self.get_eightk()
            metadata = self._eightk_metadata.to_dict() if self._eightk_metadata else {}

            if not eightk.has_earnings:
                return {
                    "has_earnings": False,
                    "reason": "Latest 8-K does not contain Item 2.02 earnings or lacks parseable EX-99.1",
                    "metadata": metadata,
                }

            earnings = eightk.earnings
            result: Dict[str, Any] = {
                "has_earnings": True,
                "detected_scale": str(earnings.detected_scale),
                "context": earnings.to_context(detail="standard"),
                "metadata": metadata,
            }

            # Each financial table is optional — some press releases omit them
            if earnings.income_statement:
                result["income_statement"] = json.loads(
                    earnings.income_statement.dataframe.to_json(orient="split")
                )
            if earnings.balance_sheet:
                result["balance_sheet"] = json.loads(
                    earnings.balance_sheet.dataframe.to_json(orient="split")
                )
            if earnings.cash_flow_statement:
                result["cash_flow"] = json.loads(
                    earnings.cash_flow_statement.dataframe.to_json(orient="split")
                )

            return result
        except Exception as e:
            return {"has_earnings": False, "reason": f"Error parsing earnings: {e}", "metadata": {}}

    def get_section(self, form: Literal["10-K", "10-Q", "20-F"], item: str) -> Dict[str, Any]:
        """Extract one section from a 10-K, 10-Q, or 20-F filing.

        Returns {"text", "metadata", "found"} where metadata includes the
        accession_number — the stable key for future database caching.
        The DB layer will wrap this method, not the tool layer above it.

        Item codes:
          10-K: "1" business, "1A" risk factors, "1C" cybersecurity,
                "3" legal proceedings, "7" MD&A, "7A" market risk
          10-Q: "1A" risk factors, "2" MD&A, "3" market risk,
                "4" controls, "5" other information
          20-F: "1A" risk factors (via .risk_factors),
                "7" MD&A (via .management_discussion)
        """
        try:
            if form == "20-F":
                return self._get_twentyf_section(item)
            elif form == "10-K":
                filing_obj = self.get_tenk()
                metadata = self._tenk_metadata
                key = _TENK_ITEM_KEYS.get(item)
            else:
                filing_obj = self.get_tenq()
                metadata = self._tenq_metadata
                key = _TENQ_ITEM_KEYS.get(item)

            if key is None:
                return {
                    "text": f"Item '{item}' not supported for {form}",
                    "metadata": metadata.to_dict() if metadata else {},
                    "found": False,
                }

            text = filing_obj[key]
            found = text is not None and bool(str(text).strip())
            if not found:
                text = f"Section {key} not found in {form} filing"

            return {
                "text": str(text) if text else "",
                "metadata": metadata.to_dict() if metadata else {},
                "found": found,
            }

        except Exception as e:
            return {"text": f"Error extracting {item} from {form}: {e}", "metadata": {}, "found": False}

    def _get_twentyf_section(self, item: str) -> Dict[str, Any]:
        """Extract a section from a 20-F filing using named accessors.

        edgartools' TwentyF class uses property accessors rather than
        bracket-style item keys. We map the same item codes used for
        10-K (e.g., "1A" for risk factors, "7" for MD&A) to the
        corresponding TwentyF properties.
        """
        # Map 10-K item codes to TwentyF named properties
        _TWENTYF_ACCESSOR_MAP = {
            "1A": "risk_factors",
            "7": "management_discussion",
            "1": "business",
        }

        accessor = _TWENTYF_ACCESSOR_MAP.get(item)
        if accessor is None:
            return {
                "text": f"Item '{item}' not supported for 20-F",
                "metadata": self._twentyf_metadata.to_dict() if self._twentyf_metadata else {},
                "found": False,
            }

        try:
            obj = self.get_twentyf()
            metadata = self._twentyf_metadata
            text = getattr(obj, accessor, None)
            found = text is not None and bool(str(text).strip())

            return {
                "text": str(text) if found else f"Section {accessor} not found in 20-F filing",
                "metadata": metadata.to_dict() if metadata else {},
                "found": found,
            }
        except Exception as e:
            return {"text": f"Error extracting {item} from 20-F: {e}", "metadata": {}, "found": False}

    def get_mda_raw(self, form: Literal["10-K", "10-Q", "20-F"] = "10-K") -> Dict[str, Any]:
        """MD&A: Item 7 from 10-K/20-F, Item 2 from 10-Q."""
        return self.get_section(form, "7" if form in ("10-K", "20-F") else "2")

    def get_risk_factors_raw(self, form: Literal["10-K", "10-Q", "20-F"] = "10-K") -> Dict[str, Any]:
        """Risk Factors: Item 1A in all forms."""
        return self.get_section(form, "1A")

    def get_business_raw(self) -> Dict[str, Any]:
        """Company Business overview — 10-K Item 1."""
        return self.get_section("10-K", "1")

    def get_cybersecurity_raw(self) -> Dict[str, Any]:
        """Cybersecurity risk management — 10-K Item 1C (SEC-mandated since 2023)."""
        return self.get_section("10-K", "1C")

    def get_legal_proceedings_raw(self) -> Dict[str, Any]:
        """Legal Proceedings — 10-K Item 3."""
        return self.get_section("10-K", "3")

    def get_market_risk_raw(self, form: Literal["10-K", "10-Q"] = "10-K") -> Dict[str, Any]:
        """Market Risk: Item 7A from 10-K, Item 3 from 10-Q."""
        return self.get_section(form, "7A" if form == "10-K" else "3")

    def get_balance_sheet(
        self, form: Literal["10-K", "10-Q", "both"] = "both"
    ) -> Dict[str, Any]:
        """Get balance sheet data with metadata."""
        return self.extract_balance_sheet_as_json(form)

    def get_income_statement(self, form: Literal["10-K", "10-Q"] = "10-K") -> Optional[dict]:
        """Income statement from 10-K or 10-Q as a JSON-serializable dict.

        Returns the XBRL-parsed income statement DataFrame in ``orient="split"``
        format, or ``None`` if the filing is unavailable or lacks XBRL data.
        """
        try:
            obj = self.get_tenk() if form == "10-K" else self.get_tenq()
            stmt = obj.financials.income_statement()
            if stmt is None:
                return None
            return json.loads(stmt.to_dataframe().to_json(orient="split"))
        except Exception:
            return None

    def get_cashflow_statement(self, form: Literal["10-K", "10-Q"] = "10-K") -> Optional[dict]:
        """Cash flow statement from 10-K or 10-Q as a JSON-serializable dict.

        Returns the XBRL-parsed cash flow statement DataFrame in ``orient="split"``
        format, or ``None`` if the filing is unavailable or lacks XBRL data.
        """
        try:
            obj = self.get_tenk() if form == "10-K" else self.get_tenq()
            stmt = obj.financials.cashflow_statement()
            if stmt is None:
                return None
            return json.loads(stmt.to_dataframe().to_json(orient="split"))
        except Exception:
            return None

    # Legacy methods for backward compatibility (updated to be form-aware)
    def extract_risk_factors(self, form: Literal["10-K", "10-Q"] = "10-K") -> str:
        """Extract risk factors from specified filing form."""
        print(f"Extracting risk factors from {form}")
        result = self.get_section(form, "1A")
        return result["text"]

    def extract_management_discussion(
        self, form: Literal["10-K", "10-Q"] = "10-K"
    ) -> str:
        """Extract management discussion from specified filing form."""
        print(f"Extracting management discussion from {form}")
        if form == "10-K":
            result = self.get_section(form, "7")
        else:  # 10-Q
            result = self.get_section(form, "2")
        return result["text"]

    def extract_balance_sheet_as_str(
        self, which: Literal["tenk", "tenq", "both", "10-K", "10-Q"] = "both"
    ) -> dict[str, str]:
        print("Extracting balance sheet as string")
        out: dict[str, str] = {}

        # Handle both old and new parameter formats
        include_tenk = which in ("tenk", "both", "10-K")
        include_tenq = which in ("tenq", "both", "10-Q")

        if include_tenk:
            tenk = self.get_tenk()
            out["tenk"] = (
                tenk.financials.balance_sheet().to_dataframe().to_string()
            )
        if include_tenq:
            try:
                tenq = self.get_tenq()
                out["tenq"] = (
                    tenq.financials.balance_sheet().to_dataframe().to_string()
                )
            except Exception as e:
                out["tenq_error"] = str(e)
        return out

    def extract_balance_sheet_as_json(
        self, which: Literal["tenk", "tenq", "both", "10-K", "10-Q"] = "both"
    ) -> dict:
        """
        Extract balance sheets from 10-K and/or 10-Q filings and convert to JSON-serializable format.
        Includes filing metadata for provenance.
        """
        print("Extracting balance sheet as JSON")
        out: dict[str, Optional[dict]] = {}

        # Handle both old and new parameter formats
        include_tenk = which in ("tenk", "both", "10-K")
        include_tenq = which in ("tenq", "both", "10-Q")

        if include_tenk:
            tenk = self.get_tenk()
            out["tenk"] = json.loads(
                tenk.financials.balance_sheet()
                .to_dataframe()
                .to_json(orient="split")
            )
            # Add metadata for provenance
            if self._tenk_metadata:
                out["tenk_metadata"] = self._tenk_metadata.to_dict()

        if include_tenq:
            try:
                tenq = self.get_tenq()
                out["tenq"] = json.loads(
                    tenq.financials.balance_sheet()
                    .to_dataframe()
                    .to_json(orient="split")
                )
                # Add metadata for provenance
                if self._tenq_metadata:
                    out["tenq_metadata"] = self._tenq_metadata.to_dict()
            except Exception as e:
                out["tenq_error"] = str(e)
                if self._tenq_metadata:
                    out["tenq_metadata"] = self._tenq_metadata.to_dict()
                else:
                    out["tenq_metadata"] = {
                        "error": "Filing information unavailable due to error"
                    }
        return out
