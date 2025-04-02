from edgar import Company, set_identity
from dotenv import load_dotenv
import os


class SECDataRetrieval:
    def __init__(self, ticker: str):
        load_dotenv()
        set_identity(os.getenv("SEC_HEADER"))
        try:
            self.company = Company(ticker)
        except Exception as e:
            print(f"Failed to initialize company in SECDataRetrieval: {e}")
            raise
        self.tenk = self._fetch_company_tenk_filing()
        self.tenq = self._fetch_company_tenq_filing()

    # Private method for internal use
    def _fetch_company_tenk_filing(self):
        return self.company.latest(form="10-K").obj()

    def _fetch_company_tenq_filing(self):
        return self.company.latest(form="10-Q").obj()

    def extract_balance_sheet_as_str(self) -> dict[str, str]:
        return {
            "tenk": self.tenk.financials.get_balance_sheet()
            .get_dataframe()
            .to_string(),
            "tenq": self.tenq.financials.get_balance_sheet()
            .get_dataframe()
            .to_string(),
        }

    def extract_risk_factors(self) -> str:
        """Extract risk factors from 10-K filing only."""
        return self.tenk.risk_factors

    def extract_management_discussion(self) -> str:
        return self.tenk.management_discussion
