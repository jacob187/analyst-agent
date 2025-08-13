from edgar import Company, set_identity
from dotenv import load_dotenv
import os
import json
from typing import Literal, Optional


class SECDataRetrieval:
    def __init__(self, ticker: str):
        load_dotenv()
        set_identity(os.getenv("SEC_HEADER"))
        try:
            self.ticker = ticker
            self.company = Company(ticker)
            # Lazy caches
            self._tenk = None
            self._tenq = None
        except Exception as e:
            print(f"Failed to initialize company in SECDataRetrieval: {e}")
            raise

    # Lazy getters
    def get_tenk(self):
        if self._tenk is None:
            self._tenk = self._fetch_company_tenk_filing()
            if self._tenk is None:
                raise ValueError(f"No 10-K available for {self.ticker}")
        return self._tenk

    def get_tenq(self):
        if self._tenq is None:
            self._tenq = self._fetch_company_tenq_filing()
            if self._tenq is None:
                raise ValueError(f"No 10-Q available for {self.ticker}")
        return self._tenq

    # Private fetchers
    def _fetch_company_tenk_filing(self):
        filing = self.company.latest(form="10-K")
        if filing is None:
            print(f"No 10-K filing found for {self.company.name}")
            return None
        print("10-K filing found")
        return filing.obj()

    def _fetch_company_tenq_filing(self):
        filing = self.company.latest(form="10-Q")
        if filing is None:
            print(f"No 10-Q filing found for {self.company.name}")
            return None
        print("10-Q filing found")
        return filing.obj()

    def extract_balance_sheet_as_str(
        self, which: Literal["tenk", "tenq", "both"] = "both"
    ) -> dict[str, str]:
        print("Extracting balance sheet as string")
        out: dict[str, str] = {}
        if which in ("tenk", "both"):
            tenk = self.get_tenk()
            out["tenk"] = (
                tenk.financials.get_balance_sheet().get_dataframe().to_string()
            )
        if which in ("tenq", "both"):
            try:
                tenq = self.get_tenq()
                out["tenq"] = (
                    tenq.financials.get_balance_sheet().get_dataframe().to_string()
                )
            except Exception as e:
                out["tenq_error"] = str(e)
        return out

    def extract_balance_sheet_as_json(
        self, which: Literal["tenk", "tenq", "both"] = "both"
    ) -> dict:
        """
        Extract balance sheets from 10-K and/or 10-Q filings and convert to JSON-serializable format.
        """
        print("Extracting balance sheet as JSON")
        out: dict[str, Optional[dict]] = {}
        if which in ("tenk", "both"):
            tenk = self.get_tenk()
            out["tenk"] = json.loads(
                tenk.financials.get_balance_sheet()
                .get_dataframe()
                .to_json(orient="split")
            )
        if which in ("tenq", "both"):
            try:
                tenq = self.get_tenq()
                out["tenq"] = json.loads(
                    tenq.financials.get_balance_sheet()
                    .get_dataframe()
                    .to_json(orient="split")
                )
            except Exception as e:
                out["tenq_error"] = str(e)
        return out

    def extract_risk_factors(self) -> str:
        """Extract risk factors from 10-K filing only."""
        print("Extracting risk factors")
        return self.get_tenk().risk_factors

    def extract_management_discussion(self) -> str:
        print("Extracting management discussion")
        return self.get_tenk().management_discussion
