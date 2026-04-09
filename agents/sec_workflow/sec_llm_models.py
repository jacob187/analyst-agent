from typing import List, Optional, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from agents.prompts import (
    MDA_ANALYSIS_SYSTEM_PROMPT,
    MDA_ANALYSIS_USER_TEMPLATE,
    RISK_FACTORS_SYSTEM_PROMPT,
    RISK_FACTORS_USER_TEMPLATE,
    BALANCE_SHEET_SYSTEM_PROMPT,
    BALANCE_SHEET_USER_TEMPLATE,
    EARNINGS_ANALYSIS_SYSTEM_PROMPT,
    EARNINGS_ANALYSIS_USER_TEMPLATE,
    MATERIAL_EVENT_SYSTEM_PROMPT,
    MATERIAL_EVENT_USER_TEMPLATE,
)


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


class EarningsAnalysis(BaseModel):
    """LLM analysis of 8-K Item 2.02 earnings release.

    Input comes from ``SECDataRetrieval.get_earnings_data()`` which provides
    structured financial tables (income, balance sheet, cash flow) parsed from
    the EX-99.1 press release exhibit via edgartools ``EarningsRelease``.
    The LLM interprets the numbers rather than extracting them.
    """

    summary: str = Field(description="Concise summary of earnings results")
    key_metrics: List[str] = Field(
        description="Key financial metrics (revenue, EPS, margins, growth rates)"
    )
    beats_misses: List[str] = Field(
        description="Notable beats or misses vs prior period or guidance"
    )
    guidance: Optional[str] = Field(
        None, description="Forward guidance if provided in the release"
    )
    sentiment_score: float = Field(
        description="Sentiment score from -10 (very negative) to 10 (very positive)"
    )
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    filing_metadata: Optional[Dict[str, str]] = Field(
        None, description="Filing metadata including dates and periods"
    )


class MaterialEventAnalysis(BaseModel):
    """LLM analysis of non-earnings 8-K material events.

    Covers items like 1.01 (material agreements), 5.02 (leadership changes),
    1.05 (cybersecurity incidents), 7.01 (Regulation FD / forward guidance).
    Input is narrative text from ``EightK.__getitem__``.
    """

    summary: str = Field(description="Concise summary of the material event")
    event_type: str = Field(
        description="Event classification (e.g. agreement, leadership_change, cybersecurity)"
    )
    key_points: List[str] = Field(description="Key details of the event")
    impact_assessment: str = Field(
        description="Potential impact on the company and its stakeholders"
    )
    sentiment_score: float = Field(
        description="Sentiment score from -10 (very negative) to 10 (very positive)"
    )
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    filing_metadata: Optional[Dict[str, str]] = Field(
        None, description="Filing metadata including dates and periods"
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
        self.earnings_parser = PydanticOutputParser(pydantic_object=EarningsAnalysis)
        self.material_event_parser = PydanticOutputParser(
            pydantic_object=MaterialEventAnalysis
        )

    def generate_mda_prompt(
        self, ticker: str, mda_data: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """Generate a prompt for analyzing Management Discussion and Analysis from specified form."""
        form_type = mda_data.get("metadata", {}).get("form", "Unknown")

        system_message = MDA_ANALYSIS_SYSTEM_PROMPT.format(form_type=form_type)

        filing_info = mda_data.get("metadata", {})
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        user_template = MDA_ANALYSIS_USER_TEMPLATE

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

        system_message = RISK_FACTORS_SYSTEM_PROMPT.format(form_type=form_type)

        filing_info = risk_data.get("metadata", {})
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        user_template = RISK_FACTORS_USER_TEMPLATE

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

        system_message = BALANCE_SHEET_SYSTEM_PROMPT
        user_template = BALANCE_SHEET_USER_TEMPLATE

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

    # ── 8-K analysis methods ──────────────────────────────────────────────

    def generate_earnings_prompt(
        self, ticker: str, earnings_data: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """Build prompt for 8-K earnings analysis.

        ``earnings_data`` comes from ``SECDataRetrieval.get_earnings_data()`` —
        contains ``context`` (edgartools' token-efficient summary), optional
        financial table JSON, and filing metadata.
        """
        filing_info = earnings_data.get("metadata", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", EARNINGS_ANALYSIS_SYSTEM_PROMPT),
            ("user", EARNINGS_ANALYSIS_USER_TEMPLATE),
        ])

        return prompt.partial(
            ticker=ticker,
            filing_date=filing_info.get("filing_date", "Unknown"),
            period=filing_info.get("period_of_report", "Unknown"),
            earnings_context=earnings_data.get("context", ""),
            detected_scale=earnings_data.get("detected_scale", "Unknown"),
            income_statement=str(earnings_data.get("income_statement", "Not available")),
            balance_sheet=str(earnings_data.get("balance_sheet", "Not available")),
            cash_flow=str(earnings_data.get("cash_flow", "Not available")),
            format_instructions=self.earnings_parser.get_format_instructions(),
        )

    def generate_material_event_prompt(
        self, ticker: str, event_data: Dict[str, Any]
    ) -> ChatPromptTemplate:
        """Build prompt for non-earnings 8-K material event analysis.

        ``event_data`` should contain ``content_type``, ``items``, ``context``,
        and the raw ``text`` of the primary item being analyzed.
        """
        filing_info = event_data.get("metadata", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", MATERIAL_EVENT_SYSTEM_PROMPT),
            ("user", MATERIAL_EVENT_USER_TEMPLATE),
        ])

        return prompt.partial(
            ticker=ticker,
            filing_date=filing_info.get("filing_date", "Unknown"),
            period=filing_info.get("period_of_report", "Unknown"),
            content_type=event_data.get("content_type", "other"),
            items=str(event_data.get("items", [])),
            event_context=event_data.get("context", ""),
            event_text=event_data.get("text", ""),
            format_instructions=self.material_event_parser.get_format_instructions(),
        )

    def analyze_earnings(
        self, ticker: str, earnings_data: Dict[str, Any]
    ) -> EarningsAnalysis:
        """Analyze 8-K earnings release with structured financial data."""
        prompt = self.generate_earnings_prompt(ticker, earnings_data)
        try:
            chain = prompt | self.llm | self.earnings_parser
            return chain.invoke({})
        except Exception as e:
            print(f"Error processing earnings: {e}")
            return EarningsAnalysis(
                summary="Error analyzing earnings release.",
                key_metrics=["Unable to extract key metrics."],
                beats_misses=["Unable to determine beats or misses."],
                guidance=None,
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                filing_metadata=earnings_data.get("metadata", {}),
            )

    def analyze_material_event(
        self, ticker: str, event_data: Dict[str, Any]
    ) -> MaterialEventAnalysis:
        """Analyze non-earnings 8-K material event."""
        prompt = self.generate_material_event_prompt(ticker, event_data)
        try:
            chain = prompt | self.llm | self.material_event_parser
            return chain.invoke({})
        except Exception as e:
            print(f"Error processing material event: {e}")
            return MaterialEventAnalysis(
                summary="Error analyzing material event.",
                event_type=event_data.get("content_type", "unknown"),
                key_points=["Unable to extract key points."],
                impact_assessment="Analysis unavailable due to processing error.",
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                filing_metadata=event_data.get("metadata", {}),
            )
