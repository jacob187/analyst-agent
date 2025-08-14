from typing import List, Optional, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Pydantic models for structured output
class SentimentAnalysis(BaseModel):
    """Model for sentiment analysis of text."""

    score: float = Field(
        description="Sentiment score from -10 (very negative) to 10 (very positive)"
    )
    analysis: str = Field(description="Detailed explanation of the sentiment score")


class MDnAAnalysis(BaseModel):
    """Full analysis for Management Discussion and Analysis"""

    summary: str = Field(description="Concise summary of the MD&A section")
    key_points: List[str] = Field(description="List of key points from MD&A")
    financial_highlights: List[str] = Field(
        description="Key financial metrics and trends discussed"
    )
    future_outlook: str = Field(description="Management's outlook for the future")
    sentiment_score: float = Field(description="Overall sentiment score")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    form_type: str = Field(description="Form type (10-K or 10-Q)")
    filing_metadata: Optional[Dict[str, str]] = Field(
        None, description="Filing metadata including dates and periods"
    )
    comparison: Optional[str] = Field(
        None, description="Comparison between 10-K and 10-Q if both are available"
    )


class RiskFactorAnalysis(BaseModel):
    """Full analysis for Risk Factors section"""

    summary: str = Field(description="Concise summary of risk factors")
    key_risks: List[str] = Field(description="List of key risk factors")
    risk_categories: Dict[str, List[str]] = Field(
        description="Risk factors categorized by type"
    )
    sentiment_score: float = Field(description="Overall risk severity score")
    sentiment_analysis: str = Field(description="Detailed risk assessment explanation")
    form_type: str = Field(description="Form type (10-K or 10-Q)")
    filing_metadata: Optional[Dict[str, str]] = Field(
        None, description="Filing metadata including dates and periods"
    )
    comparison: Optional[str] = Field(
        None, description="Comparison between 10-K and 10-Q if both are available"
    )


class BalanceSheetAnalysis(BaseModel):
    """Full analysis for Balance Sheet section"""

    summary: str = Field(description="Concise summary of balance sheet")
    key_metrics: List[str] = Field(description="List of key metrics from balance sheet")
    liquidity_analysis: str = Field(description="Liquidity analysis of balance sheet")
    solvency_analysis: str = Field(description="Solvency analysis of balance sheet")
    growth_trends: str = Field(description="Growth trends in balance sheet")
    financial_highlights: List[str] = Field(
        description="Key data in the balance sheet that has larger impact with respect to the other data"
    )
    red_flags: List[str] = Field(description="Red flags in the balance sheet")
    comparison: Optional[str] = Field(
        None, description="Comparison between 10-K and 10-Q if both are available"
    )


class SECDocumentProcessor:
    """Processes SEC documents using LLM."""

    def __init__(self, llm: BaseChatModel):
        """Initialize with OpenAI API key."""

        self.llm = llm

        # Initialize output parsers
        self.mda_parser = PydanticOutputParser(pydantic_object=MDnAAnalysis)
        self.risk_parser = PydanticOutputParser(pydantic_object=RiskFactorAnalysis)
        self.balance_sheet_parser = PydanticOutputParser(
            pydantic_object=BalanceSheetAnalysis
        )

    def generate_mda_prompt(
        self, ticker: str, mda_data: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """Generate a prompt for analyzing Management Discussion and Analysis from specified form."""
        form_type = mda_data.get("metadata", {}).get("form", "Unknown")

        system_message = f"""You are a financial expert analyzing the Management Discussion and Analysis (MD&A) 
        section from a {form_type} SEC filing. Provide a comprehensive analysis including a summary, key points, 
        financial highlights, future outlook, and sentiment analysis.
        
        IMPORTANT: Include the form type and filing metadata in your analysis to provide proper provenance.
        You must respond with a properly formatted JSON object that matches the schema exactly.
        DO NOT return the schema definition - fill in actual values based on your analysis.
        """

        filing_info = mda_data.get("metadata", {})
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        user_template = """Analyze the following MD&A section from {ticker}'s {form_type} SEC filing:
        
        Filing Date: {filing_date}
        Period of Report: {period}
        
        MD&A Content:
        {mda_text}
        
        Follow this JSON schema EXACTLY and fill in the values with your analysis:
        {format_instructions}
        
        Your response should be a valid JSON object with real values, not placeholders or field descriptions.
        Make sure to include the form_type and filing_metadata fields with the provided information."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", user_template),
            ]
        )

        return prompt.partial(
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            period=period,
            mda_text=mda_data.get("text", ""),
            format_instructions=self.mda_parser.get_format_instructions(),
        )

    def generate_risk_factors_prompt(
        self, ticker: str, risk_data: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """Generate a prompt for analyzing Risk Factors from specified form."""
        form_type = risk_data.get("metadata", {}).get("form", "Unknown")

        system_message = f"""You are a financial risk analyst examining the Risk Factors section from a {form_type} SEC filing.
        Provide a comprehensive analysis including a summary, key risks, risk categorization,
        and an overall assessment of risk severity.
        
        Note: If this is a 10-Q filing, Risk Factors may only include material changes since the last 10-K,
        or may not be present at all if there are no material changes.
        
        IMPORTANT: Include the form type and filing metadata in your analysis to provide proper provenance.
        You must respond with a properly formatted JSON object that matches the schema exactly.
        DO NOT return the schema definition - fill in actual values based on your analysis.
        """

        filing_info = risk_data.get("metadata", {})
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        user_template = """Analyze the following Risk Factors section from {ticker}'s {form_type} SEC filing:
        
        Filing Date: {filing_date}
        Period of Report: {period}
        
        Risk Factors Content:
        {risk_text}
        
        Follow this JSON schema EXACTLY and fill in the values with your analysis:
        {format_instructions}
        
        Your response should be a valid JSON object with real values, not placeholders or field descriptions.
        Make sure to include the form_type and filing_metadata fields with the provided information."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", user_template),
            ]
        )

        return prompt.partial(
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            period=period,
            risk_text=risk_data.get("text", ""),
            format_instructions=self.risk_parser.get_format_instructions(),
        )

    def generate_balance_sheet_prompt(
        self,
        ticker: str,
        tenk: dict,
        tenq: dict,
    ) -> ChatPromptTemplate:
        """Generate a prompt for analyzing Balance Sheet from 10-K and 10-Q."""

        system_message = """You are a financial expert analyzing the Balance Sheet section of SEC filings. 
        Provide a comprehensive analysis including a summary, key points, financial highlights, 
        and comparison between 10-K and 10-Q if both are available.
            
            IMPORTANT: You must respond with a properly formatted JSON object that matches the schema exactly.
            DO NOT return the schema definition - fill in actual values based on your analysis.
            """

        user_template = """Analyze the following Balance Sheet section from {ticker}'s 10-K and 10-Q SEC filings:
            
            {tenk}
            
            {tenq}
            
            Follow this JSON schema EXACTLY and fill in the values with your analysis:
            {format_instructions}
            
            Your response should be a valid JSON object with real values, not placeholders or field descriptions."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", user_template),
            ]
        )

        return prompt.partial(
            ticker=ticker,
            tenk=tenk,
            tenq=tenq,
            format_instructions=self.balance_sheet_parser.get_format_instructions(),
        )

    def analyze_balance_sheet(
        self, ticker: str, tenk: dict, tenq: dict
    ) -> BalanceSheetAnalysis:
        prompt = self.generate_balance_sheet_prompt(ticker, tenk, tenq)
        try:
            # Use the Langchain chain approach
            chain = prompt | self.llm | self.balance_sheet_parser
            return chain.invoke({})
        except Exception as e:
            print(f"Error processing balance sheet: {e}")
            # Return fallback values
            return BalanceSheetAnalysis(
                ticker=ticker,
                summary="Error analyzing balance sheet section.",
                key_metrics=["Unable to extract key metrics."],
                liquidity_analysis="Analysis unavailable due to processing error.",
                solvency_analysis="Analysis unavailable due to processing error.",
                growth_trends="Analysis unavailable due to processing error.",
                financial_highlights=["Unable to extract financial highlights."],
                red_flags=["Data processing error."],
                comparison=None,
            )

    def analyze_mda(self, ticker: str, mda_data: Dict[str, Any]) -> MDnAAnalysis:
        """Analyze the Management Discussion section from specified form."""
        prompt = self.generate_mda_prompt(ticker, mda_data)
        try:
            # Use the Langchain chain approach
            chain = prompt | self.llm | self.mda_parser
            return chain.invoke({})
        except Exception as e:
            print(f"Error processing MD&A: {e}")
            # Return fallback values
            return MDnAAnalysis(
                summary="Error analyzing MD&A section.",
                key_points=["Unable to extract key points."],
                financial_highlights=["Unable to extract financial highlights."],
                future_outlook="Analysis unavailable due to processing error.",
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                form_type=mda_data.get("metadata", {}).get("form", "Unknown"),
                filing_metadata=mda_data.get("metadata", {}),
                comparison=None,
            )

    def analyze_risk_factors(
        self, ticker: str, risk_data: Dict[str, Any]
    ) -> RiskFactorAnalysis:
        """Analyze the Risk Factors section from specified form."""
        prompt = self.generate_risk_factors_prompt(ticker, risk_data)
        try:
            # Use the Langchain chain approach
            chain = prompt | self.llm | self.risk_parser
            return chain.invoke({})
        except Exception as e:
            print(f"Error processing Risk Factors: {e}")
            # Return fallback values
            return RiskFactorAnalysis(
                summary="Error analyzing Risk Factors section.",
                key_risks=["Unable to extract key risks."],
                risk_categories={"Processing Error": ["Unable to categorize risks."]},
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                form_type=risk_data.get("metadata", {}).get("form", "Unknown"),
                filing_metadata=risk_data.get("metadata", {}),
                comparison=None,
            )
