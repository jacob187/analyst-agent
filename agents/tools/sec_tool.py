from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any

from agents.sec_workflow.get_SEC_data import SECDataRetrieval
from agents.sec_workflow.sec_llm_models import (
    SECDocumentProcessor,
    MDnAAnalysis,
    RiskFactorAnalysis,
    BalanceSheetAnalysis,
)


class SECDataTools:
    """
    A class that provides separate tools for retrieving different types of SEC filing data.
    Each tool uses a shared data_retriever instance for efficiency.
    """

    def __init__(self, ticker: str):
        """
        Initialize with a ticker and create a data_retriever object.

        Args:
            ticker: The company ticker symbol (e.g., 'AAPL').
        """
        self.ticker = ticker
        self.data_retriever = SECDataRetrieval(ticker)

    @tool
    def get_management_discussion(self) -> str:
        """
        Fetches the Management Discussion & Analysis (MD&A) text from the latest SEC filings.

        Returns:
            The MD&A text or an error message.
        """
        try:
            return self.data_retriever.extract_management_discussion()
        except Exception as e:
            return f"Failed to retrieve MD&A data for {self.ticker}: {e}"

    @tool
    def get_risk_factors(self) -> str:
        """
        Fetches the Risk Factors text from the latest SEC filings.

        Returns:
            The Risk Factors text or an error message.
        """
        try:
            return self.data_retriever.extract_risk_factors()
        except Exception as e:
            return f"Failed to retrieve Risk Factors data for {self.ticker}: {e}"

    @tool
    def get_balance_sheets(self) -> Dict[str, Any]:
        """
        Fetches the balance sheet data from the latest 10-K and 10-Q filings.

        Returns:
            A dictionary containing the balance sheets or an error message.
        """
        try:
            return self.data_retriever.extract_balance_sheet_as_json()
        except Exception as e:
            return {
                "error": f"Failed to retrieve balance sheet data for {self.ticker}: {e}"
            }

    @tool
    def get_all_sec_data(self) -> Dict[str, Any]:
        """
        Fetches all key data from the latest SEC filings (MD&A, Risk Factors, and balance sheets).

        Returns:
            A dictionary containing all extracted data or an error message.
        """
        try:
            mda_text = self.data_retriever.extract_management_discussion()
            risk_text = self.data_retriever.extract_risk_factors()
            balance_sheets_json = self.data_retriever.extract_balance_sheet_as_json()

            return {
                "ticker": self.ticker,
                "mda_text": mda_text,
                "risk_text": risk_text,
                "balance_sheets_json": balance_sheets_json,
            }
        except Exception as e:
            return {"error": f"Failed to retrieve SEC data for {self.ticker}: {e}"}


# Legacy function for backward compatibility
@tool
def get_sec_filing_data(ticker: str) -> Dict[str, Any]:
    """
    Fetches key data from the latest 10-K and 10-Q filings for a given company ticker.
    Uses SECDataRetrieval to get MD&A text, Risk Factors text, and balance sheets.

    Args:
        ticker: The company ticker symbol (e.g., 'AAPL').

    Returns:
        A dictionary containing the extracted data or an error message.
    """
    try:
        data_retriever = SECDataRetrieval(ticker)

        # Extract all necessary data at once
        mda_text = data_retriever.extract_management_discussion()
        risk_text = data_retriever.extract_risk_factors()
        balance_sheets_json = data_retriever.extract_balance_sheet_as_json()

        return {
            "ticker": ticker,
            "mda_text": mda_text,
            "risk_text": risk_text,
            "balance_sheets_json": balance_sheets_json,
        }
    except Exception as e:
        return {"error": f"Failed to retrieve SEC data for {ticker}: {e}"}


class SECAnalysisTools:
    """
    A factory class to create SEC analysis tools that share an LLM instance.
    Uses the existing SECDocumentProcessor for analysis logic.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize with an LLM that will be used for all analysis tools.

        Args:
            llm: A LangChain chat model instance.
        """
        self.document_processor = SECDocumentProcessor(llm)

    @tool
    def analyze_summary_management_discussion(
        self, mda_text: str, ticker: str
    ) -> MDnAAnalysis:
        """
        Analyzes the Management Discussion & Analysis (MD&A) section of an SEC filing.
        Uses the existing SECDocumentProcessor logic.

        Args:
            mda_text: The full text of the MD&A section.
            ticker: The company ticker symbol.

        Returns:
            An MDnAAnalysis object with the structured analysis.
        """
        return self.document_processor.analyze_mda(ticker, mda_text)

    @tool
    def analyze_summary_risk_factors(
        self, risk_text: str, ticker: str
    ) -> RiskFactorAnalysis:
        """
        Analyzes the Risk Factors section of an SEC filing.
        Uses the existing SECDocumentProcessor logic.

        Args:
            risk_text: The full text of the Risk Factors section.
            ticker: The company ticker symbol.

        Returns:
            A RiskFactorAnalysis object with the structured analysis.
        """
        return self.document_processor.analyze_risk_factors(ticker, risk_text)

    @tool
    def analyze_summary_balance_sheet(
        self, balance_sheets_json: dict, ticker: str
    ) -> BalanceSheetAnalysis:
        """
        Analyzes the balance sheet data from 10-K and 10-Q filings.
        Uses the existing SECDocumentProcessor logic.

        Args:
            balance_sheets_json: A dictionary containing the 10-K and 10-Q balance sheets.
            ticker: The company ticker symbol.

        Returns:
            A BalanceSheetAnalysis object with the structured analysis.
        """
        return self.document_processor.analyze_balance_sheet(
            ticker,
            balance_sheets_json.get("tenk", {}),
            balance_sheets_json.get("tenq", {}),
        )
