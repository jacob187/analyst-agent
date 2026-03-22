from edgar import Company, set_identity
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
        self.company = Company(ticker)
        # Lazy caches
        self._tenk_filing = None
        self._tenq_filing = None
        self._tenk_obj = None
        self._tenq_obj = None
        self._tenk_metadata = None
        self._tenq_metadata = None

    def check_filing_availability(self) -> Dict[str, Any]:
        """Check what filings are available for this company.

        Returns:
            Dict with ticker, company_name, has_10k, and has_10q fields.
        """
        result = {
            "ticker": self.ticker,
            "company_name": self.company.name,
            "has_10k": False,
            "has_10q": False,
        }
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

    def get_section(self, form: Literal["10-K", "10-Q"], item: str) -> Dict[str, Any]:
        """Extract one section from a 10-K or 10-Q filing.

        Returns {"text", "metadata", "found"} where metadata includes the
        accession_number — the stable key for future database caching.
        The DB layer will wrap this method, not the tool layer above it.

        Item codes:
          10-K: "1" business, "1A" risk factors, "1C" cybersecurity,
                "3" legal proceedings, "7" MD&A, "7A" market risk
          10-Q: "1A" risk factors, "2" MD&A, "3" market risk,
                "4" controls, "5" other information
        """
        try:
            if form == "10-K":
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

    def get_mda_raw(self, form: Literal["10-K", "10-Q"] = "10-K") -> Dict[str, Any]:
        """MD&A: Item 7 from 10-K, Item 2 from 10-Q."""
        return self.get_section(form, "7" if form == "10-K" else "2")

    def get_risk_factors_raw(self, form: Literal["10-K", "10-Q"] = "10-K") -> Dict[str, Any]:
        """Risk Factors: Item 1A in both forms."""
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
                tenk.financials.get_balance_sheet().get_dataframe().to_string()
            )
        if include_tenq:
            try:
                tenq = self.get_tenq()
                out["tenq"] = (
                    tenq.financials.get_balance_sheet().get_dataframe().to_string()
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
                tenk.financials.get_balance_sheet()
                .get_dataframe()
                .to_json(orient="split")
            )
            # Add metadata for provenance
            if self._tenk_metadata:
                out["tenk_metadata"] = self._tenk_metadata.to_dict()

        if include_tenq:
            try:
                tenq = self.get_tenq()
                out["tenq"] = json.loads(
                    tenq.financials.get_balance_sheet()
                    .get_dataframe()
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
