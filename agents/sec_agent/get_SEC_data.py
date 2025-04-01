from edgar import Company, set_identity
from dotenv import load_dotenv
import os

load_dotenv()

set_identity(os.getenv("SEC_HEADER"))

# Cache dictionary to store filings by ticker so we don't make multiple API requests to SEC
filing_cache = {}


def get_balance_sheet_as_dataframe(filing_obj):
    return filing_obj.financials.get_balance_sheet().get_dataframe()


def get_company_tenk_filing(ticker: str):
    # Check if we've already retrieved this filing
    if ticker in filing_cache:
        return filing_cache[ticker]

    # If not in cache, fetch it
    company = Company(ticker)
    filing = company.get_filings(form="10-K").latest().obj()
    filing_cache[ticker] = filing
    return filing


def get_risk_factors(ticker: str) -> str:
    filing = get_company_tenk_filing(ticker)
    return filing.risk_factors


def get_balance_sheet_str(ticker: str) -> str:
    filing = get_company_tenk_filing(ticker)
    return get_balance_sheet_as_dataframe(filing).to_string()
