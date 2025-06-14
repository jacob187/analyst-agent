from edgar import Company, set_identity
from dotenv import load_dotenv
import os
import json


class SECDataRetrieval:
    def __init__(self, ticker: str):
        load_dotenv()
        set_identity(os.getenv("SEC_HEADER"))
        try:
            self.company = Company(ticker)
            try:
                self.tenk = self._fetch_company_tenk_filing()
            except ValueError as e:
                print(f"Warning: {e}")
                self.tenk = None

            try:
                self.tenq = self._fetch_company_tenq_filing()
            except ValueError as e:
                print(f"Warning: {e}")
                self.tenq = None

            # If both are None, we can't proceed
            if self.tenk is None and self.tenq is None:
                raise ValueError(
                    f"No SEC filings found for ticker {ticker}. Please verify the ticker symbol."
                )

        except Exception as e:
            print(f"Failed to initialize company in SECDataRetrieval: {e}")
            raise

    # Private method for internal use
    def _fetch_company_tenk_filing(self):
        filing = self.company.latest(form="10-K")
        if filing is None:
            print(f"No 10-K filing found for {self.company.name}")
            return None
        return filing.obj()

    def _fetch_company_tenq_filing(self):
        filing = self.company.latest(form="10-Q")
        if filing is None:
            print(f"No 10-Q filing found for {self.company.name}")
            return None
        return filing.obj()

    def extract_balance_sheet_as_str(self) -> dict[str, str]:
        return {
            "tenk": self.tenk.financials.get_balance_sheet()
            .get_dataframe()
            .to_string(),
            "tenq": self.tenq.financials.get_balance_sheet()
            .get_dataframe()
            .to_string(),
        }

    def extract_balance_sheet_as_json(self) -> dict:
        """
        Extract balance sheets from both 10-K and 10-Q filings and convert to JSON-serializable format.

        Returns:
            dict: Dictionary containing structured balance sheet data for both 10-K and 10-Q
        """
        # Convert DataFrames to nested dictionaries that are JSON-serializable
        return {
            "tenk": json.loads(
                self.tenk.financials.get_balance_sheet()
                .get_dataframe()
                .to_json(orient="split")
            ),
            "tenq": json.loads(
                self.tenq.financials.get_balance_sheet()
                .get_dataframe()
                .to_json(orient="split")
            ),
        }

    def extract_risk_factors(self) -> str:
        """Extract risk factors from 10-K filing only."""
        return self.tenk.risk_factors

    def extract_management_discussion(self) -> str:
        return self.tenk.management_discussion
