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
    def analyze_management_discussion(self, mda_text: str, ticker: str) -> MDnAAnalysis:
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
    def analyze_risk_factors(self, risk_text: str, ticker: str) -> RiskFactorAnalysis:
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
    def analyze_balance_sheet(self, balance_sheets_json: dict, ticker: str) -> BalanceSheetAnalysis:
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
            balance_sheets_json.get('tenk', {}), 
            balance_sheets_json.get('tenq', {})
        )