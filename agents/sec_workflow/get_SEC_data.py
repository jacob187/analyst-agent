from edgar import Company, set_identity
from dotenv import load_dotenv
import os
import json
from typing import Literal, Optional, Dict, Any
from datetime import datetime


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
    def __init__(self, ticker: str):
        load_dotenv()
        set_identity(os.getenv("SEC_HEADER"))
        try:
            self.ticker = ticker
            self.company = Company(ticker)
            # Lazy caches
            self._tenk_filing = None
            self._tenq_filing = None
            self._tenk_obj = None
            self._tenq_obj = None
            self._tenk_metadata = None
            self._tenq_metadata = None
        except Exception as e:
            print(f"Failed to initialize company in SECDataRetrieval: {e}")
            raise

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

    def get_section(
        self, form: Literal["10-K", "10-Q"], item: Literal["1A", "2", "7"]
    ) -> Dict[str, Any]:
        """
        Extract specific sections from SEC filings with metadata.

        Args:
            form: "10-K" or "10-Q"
            item: "1A" (Risk Factors), "2" (MD&A for 10-Q), "7" (MD&A for 10-K)

        Returns:
            Dict with 'text', 'metadata', and 'found' keys
        """
        try:
            if form == "10-K":
                filing_obj = self.get_tenk()
                metadata = self._tenk_metadata
            else:  # 10-Q
                filing_obj = self.get_tenq()
                metadata = self._tenq_metadata

            text = None
            found = False

            # Use the known attributes from our test
            if form == "10-K":
                if item == "1A":
                    if hasattr(filing_obj, "risk_factors"):
                        text = filing_obj.risk_factors
                        found = True
                elif item == "7":
                    if hasattr(filing_obj, "management_discussion"):
                        text = filing_obj.management_discussion
                        found = True
            elif form == "10-Q":
                # 10-Q objects don't have risk_factors or management_discussion as convenience properties
                # We need to extract from the filing using the sections or items method
                if item == "1A":
                    text = self._extract_10q_risk_factors()
                    found = text is not None and len(text.strip()) > 0
                elif item == "2":
                    text = self._extract_10q_management_discussion()
                    found = text is not None and len(text.strip()) > 0

            if not found or not text:
                text = (
                    f"Section Item {item} not found or not available in {form} filing"
                )

            return {
                "text": text or "",
                "metadata": metadata.to_dict() if metadata else {},
                "found": found,
            }

        except Exception as e:
            print(f"Error extracting section {item} from {form}: {e}")
            return {
                "text": f"Error extracting section {item} from {form}: {e}",
                "metadata": {},
                "found": False,
            }

    def _extract_10q_risk_factors(self) -> Optional[str]:
        """
        Extract Risk Factors from 10-Q filing using the filing's items or sections.
        10-Q may not have Risk Factors if no material changes since last 10-K.
        """
        try:
            filing = self.get_tenq_filing()

            # Try to search for risk factors in the filing text
            if hasattr(filing, "search"):
                results = filing.search("Item 1A")
                if results:
                    # This is a simplified extraction - in practice you'd want more robust parsing
                    full_text = filing.text()
                    # Look for "Item 1A" section
                    import re

                    pattern = r"Item\s*1A\.?\s*Risk\s*Factors"
                    match = re.search(pattern, full_text, re.IGNORECASE)
                    if match:
                        start = match.end()
                        # Find next major item
                        next_item = re.search(
                            r"Item\s*[2-9][A-Z]?\.", full_text[start:], re.IGNORECASE
                        )
                        if next_item:
                            end = start + next_item.start()
                            return full_text[start:end].strip()
                        else:
                            # Take a reasonable chunk
                            return full_text[start : start + 5000].strip()

            return "Risk Factors section not found in 10-Q (may not be required if no material changes)"

        except Exception as e:
            print(f"Error extracting 10-Q risk factors: {e}")
            return None

    def _extract_10q_management_discussion(self) -> Optional[str]:
        """
        Extract Management Discussion and Analysis from 10-Q filing (Item 2).
        """
        try:
            filing = self.get_tenq_filing()

            # Try to search for MD&A in the filing text
            if hasattr(filing, "search"):
                full_text = filing.text()
                # Look for "Item 2" section (MD&A in 10-Q)
                import re

                pattern = r"Item\s*2\.?\s*Management.s\s*Discussion"
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    start = match.end()
                    # Find next major item
                    next_item = re.search(
                        r"Item\s*[3-9][A-Z]?\.", full_text[start:], re.IGNORECASE
                    )
                    if next_item:
                        end = start + next_item.start()
                        return full_text[start:end].strip()
                    else:
                        # Take a reasonable chunk
                        return full_text[start : start + 10000].strip()

            return "Management Discussion section not found in 10-Q"

        except Exception as e:
            print(f"Error extracting 10-Q management discussion: {e}")
            return None

    # New form-specific methods with clear naming and provenance
    def get_mda_raw(self, form: Literal["10-K", "10-Q"] = "10-K") -> Dict[str, Any]:
        """Get Management Discussion and Analysis with metadata."""
        if form == "10-K":
            return self.get_section(form, "7")
        else:
            return self.get_section(form, "2")

    def get_risk_factors_raw(
        self, form: Literal["10-K", "10-Q"] = "10-K"
    ) -> Dict[str, Any]:
        """Get Risk Factors with metadata."""
        return self.get_section(form, "1A")

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
