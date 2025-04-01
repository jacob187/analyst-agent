from edgar import Company, set_identity
from dotenv import load_dotenv
import os
import pandas as pd


class SECDataRetrieval:
    def __init__(self, ticker: str):
        load_dotenv()
        set_identity(os.getenv("SEC_HEADER"))
        try:
            self.company = Company(ticker)
        except Exception as e:
            print(f"Failed to initialize company in SECDataRetrieval: {e}")
            raise
        self.tenk = self.get_company_tenk_filing()

    def get_balance_sheet_as_dataframe(self) -> pd.DataFrame:
        return self.tenk.financials.get_balance_sheet().get_dataframe()

    def get_company_tenk_filing(self):
        return self.company.get_filings(form="10-K").latest().obj()

    def get_risk_factors(self) -> str:
        return self.tenk.risk_factors

    def get_balance_sheet_str(self) -> str:
        return self.get_balance_sheet_as_dataframe().to_string()

    def get_management_discussion(self) -> str:
        return self.tenk.management_discussion
