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
    BUSINESS_OVERVIEW_SYSTEM_PROMPT,
    BUSINESS_OVERVIEW_USER_TEMPLATE,
    CYBERSECURITY_SYSTEM_PROMPT,
    CYBERSECURITY_USER_TEMPLATE,
    LEGAL_PROCEEDINGS_SYSTEM_PROMPT,
    LEGAL_PROCEEDINGS_USER_TEMPLATE,
    MARKET_RISK_SYSTEM_PROMPT,
    MARKET_RISK_USER_TEMPLATE,
    INCOME_STATEMENT_SYSTEM_PROMPT,
    INCOME_STATEMENT_USER_TEMPLATE,
    CASH_FLOW_SYSTEM_PROMPT,
    CASH_FLOW_USER_TEMPLATE,
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
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative / high risk severity) to 10 (very positive / low risk). A filing with major risks, investigations, or material weaknesses should score strongly negative.")
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


class BusinessOverviewAnalysis(BaseModel):
    """LLM analysis of the Business Overview section (10-K Item 1)."""

    summary: str = Field(description="Concise summary of the company's business model")
    business_segments: List[str] = Field(description="Named business segments or divisions")
    key_products_services: List[str] = Field(description="Primary products or services offered")
    competitive_position: str = Field(description="Assessment of competitive positioning and moats")
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative) to 10 (very positive)")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    filing_metadata: Optional[Dict[str, str]] = Field(None, description="Filing metadata")


class CybersecurityAnalysis(BaseModel):
    """LLM analysis of the Cybersecurity Risk Management section (10-K Item 1C)."""

    summary: str = Field(description="Concise summary of cybersecurity posture and governance")
    governance_overview: str = Field(description="Board and management oversight structure")
    key_disclosures: List[str] = Field(description="Specific cybersecurity frameworks, certifications, or processes disclosed")
    red_flags: List[str] = Field(description="Disclosed incidents, material weaknesses, or vague governance concerns")
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative) to 10 (very positive)")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    filing_metadata: Optional[Dict[str, str]] = Field(None, description="Filing metadata")


class LegalProceedingsAnalysis(BaseModel):
    """LLM analysis of the Legal Proceedings section (10-K Item 3)."""

    summary: str = Field(description="Concise summary of the legal landscape")
    key_cases: List[str] = Field(description="Material legal cases or regulatory proceedings")
    red_flags: List[str] = Field(description="Proceedings with significant financial exposure or reputational risk")
    overall_risk: str = Field(description="Overall assessment of legal risk severity")
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative) to 10 (very positive)")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    filing_metadata: Optional[Dict[str, str]] = Field(None, description="Filing metadata")


class MarketRiskAnalysis(BaseModel):
    """LLM analysis of the Market Risk section (10-K Item 7A)."""

    summary: str = Field(description="Concise summary of market risk exposures")
    key_exposures: List[str] = Field(description="Identified market risk categories with quantitative details where available")
    risk_assessment: str = Field(description="Overall assessment of market risk severity and hedging adequacy")
    red_flags: List[str] = Field(description="Unhedged concentrations or material sensitivities")
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative) to 10 (very positive)")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    filing_metadata: Optional[Dict[str, str]] = Field(None, description="Filing metadata")


class IncomeStatementAnalysis(BaseModel):
    """LLM analysis of income statement data from 10-K and/or 10-Q XBRL filings."""

    summary: str = Field(description="Concise summary of income statement performance")
    key_metrics: List[str] = Field(description="Key figures: revenue, gross profit, operating income, net income, margins, growth rates")
    revenue_analysis: str = Field(description="Revenue trend and growth analysis")
    profitability_analysis: str = Field(description="Margin trends and profitability assessment")
    red_flags: List[str] = Field(description="Deteriorating trends, unusual items, or concerns")
    outlook: str = Field(description="Forward-looking assessment based on the trajectory")
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative) to 10 (very positive)")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    comparison: Optional[str] = Field(None, description="Annual vs quarterly comparison if both available")


class CashFlowAnalysis(BaseModel):
    """LLM analysis of cash flow statement data from 10-K and/or 10-Q XBRL filings."""

    summary: str = Field(description="Concise summary of cash flow generation and allocation")
    key_metrics: List[str] = Field(description="Key figures: operating cash flow, capex, free cash flow, dividends, buybacks")
    operating_cash_flow_analysis: str = Field(description="Quality and sustainability of operating cash flows")
    free_cash_flow_analysis: str = Field(description="Free cash flow generation and trend")
    red_flags: List[str] = Field(description="Negative operating cash flow, cash burn, or concerning capital allocation")
    outlook: str = Field(description="Assessment of financial flexibility and cash generation sustainability")
    sentiment_score: float = Field(description="Sentiment score from -10 (very negative) to 10 (very positive)")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    comparison: Optional[str] = Field(None, description="Annual vs quarterly comparison if both available")


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
        self.material_event_parser = PydanticOutputParser(pydantic_object=MaterialEventAnalysis)
        self.business_overview_parser = PydanticOutputParser(pydantic_object=BusinessOverviewAnalysis)
        self.cybersecurity_parser = PydanticOutputParser(pydantic_object=CybersecurityAnalysis)
        self.legal_proceedings_parser = PydanticOutputParser(pydantic_object=LegalProceedingsAnalysis)
        self.market_risk_parser = PydanticOutputParser(pydantic_object=MarketRiskAnalysis)
        self.income_statement_parser = PydanticOutputParser(pydantic_object=IncomeStatementAnalysis)
        self.cash_flow_parser = PydanticOutputParser(pydantic_object=CashFlowAnalysis)

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

    # ── New section analysis methods ──────────────────────────────────────────

    def _text_section_prompt(
        self,
        system_prompt: str,
        user_template: str,
        parser: PydanticOutputParser,
        ticker: str,
        data: Dict[str, Any],
    ) -> ChatPromptTemplate:
        """Shared prompt builder for text-based 10-K sections (Item 1, 1C, 3, 7A).

        All four sections have the same input shape: {"text": "...", "metadata": {...}}.
        """
        filing_info = data.get("metadata", {})
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_template),
        ])
        return prompt.partial(
            ticker=ticker,
            filing_date=filing_info.get("filing_date", "Unknown"),
            period=filing_info.get("period_of_report", "Unknown"),
            content_text=data.get("text", ""),
            format_instructions=parser.get_format_instructions(),
        )

    def analyze_business_overview(
        self, ticker: str, data: Dict[str, Any]
    ) -> BusinessOverviewAnalysis:
        """Analyze 10-K Item 1 — Business Overview."""
        prompt = self._text_section_prompt(
            BUSINESS_OVERVIEW_SYSTEM_PROMPT, BUSINESS_OVERVIEW_USER_TEMPLATE,
            self.business_overview_parser, ticker, data,
        )
        try:
            return (prompt | self.llm | self.business_overview_parser).invoke({})
        except Exception as e:
            print(f"Error processing Business Overview: {e}")
            return BusinessOverviewAnalysis(
                summary="Error analyzing Business Overview section.",
                business_segments=["Unable to extract segments."],
                key_products_services=["Unable to extract products/services."],
                competitive_position="Analysis unavailable due to processing error.",
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                filing_metadata=data.get("metadata", {}),
            )

    def analyze_cybersecurity(
        self, ticker: str, data: Dict[str, Any]
    ) -> CybersecurityAnalysis:
        """Analyze 10-K Item 1C — Cybersecurity Risk Management."""
        prompt = self._text_section_prompt(
            CYBERSECURITY_SYSTEM_PROMPT, CYBERSECURITY_USER_TEMPLATE,
            self.cybersecurity_parser, ticker, data,
        )
        try:
            return (prompt | self.llm | self.cybersecurity_parser).invoke({})
        except Exception as e:
            print(f"Error processing Cybersecurity: {e}")
            return CybersecurityAnalysis(
                summary="Error analyzing Cybersecurity section.",
                governance_overview="Analysis unavailable due to processing error.",
                key_disclosures=["Unable to extract disclosures."],
                red_flags=[],
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                filing_metadata=data.get("metadata", {}),
            )

    def analyze_legal_proceedings(
        self, ticker: str, data: Dict[str, Any]
    ) -> LegalProceedingsAnalysis:
        """Analyze 10-K Item 3 — Legal Proceedings."""
        prompt = self._text_section_prompt(
            LEGAL_PROCEEDINGS_SYSTEM_PROMPT, LEGAL_PROCEEDINGS_USER_TEMPLATE,
            self.legal_proceedings_parser, ticker, data,
        )
        try:
            return (prompt | self.llm | self.legal_proceedings_parser).invoke({})
        except Exception as e:
            print(f"Error processing Legal Proceedings: {e}")
            return LegalProceedingsAnalysis(
                summary="Error analyzing Legal Proceedings section.",
                key_cases=["Unable to extract cases."],
                red_flags=[],
                overall_risk="Analysis unavailable due to processing error.",
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                filing_metadata=data.get("metadata", {}),
            )

    def analyze_market_risk(
        self, ticker: str, data: Dict[str, Any]
    ) -> MarketRiskAnalysis:
        """Analyze 10-K Item 7A — Market Risk."""
        prompt = self._text_section_prompt(
            MARKET_RISK_SYSTEM_PROMPT, MARKET_RISK_USER_TEMPLATE,
            self.market_risk_parser, ticker, data,
        )
        try:
            return (prompt | self.llm | self.market_risk_parser).invoke({})
        except Exception as e:
            print(f"Error processing Market Risk: {e}")
            return MarketRiskAnalysis(
                summary="Error analyzing Market Risk section.",
                key_exposures=["Unable to extract exposures."],
                risk_assessment="Analysis unavailable due to processing error.",
                red_flags=[],
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
                filing_metadata=data.get("metadata", {}),
            )

    def _financial_statement_prompt(
        self,
        system_prompt: str,
        user_template: str,
        parser: PydanticOutputParser,
        ticker: str,
        raw_data: Dict[str, Any],
    ) -> ChatPromptTemplate:
        """Shared prompt builder for XBRL financial statement analyses.

        raw_data shape: {"tenk": <json dict or None>, "tenk_metadata": {...},
                         "tenq": <json dict or None>, "tenq_metadata": {...}}
        """
        tenk_meta = raw_data.get("tenk_metadata") or {}
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_template),
        ])
        return prompt.partial(
            ticker=ticker,
            filing_date=tenk_meta.get("filing_date", "Unknown"),
            period=tenk_meta.get("period_of_report", "Unknown"),
            tenk_data=str(raw_data.get("tenk") or "Not available"),
            tenq_data=str(raw_data.get("tenq") or "Not available"),
            format_instructions=parser.get_format_instructions(),
        )

    def analyze_income_statement(
        self, ticker: str, raw_data: Dict[str, Any]
    ) -> IncomeStatementAnalysis:
        """Analyze income statement XBRL data from 10-K (and optionally 10-Q)."""
        prompt = self._financial_statement_prompt(
            INCOME_STATEMENT_SYSTEM_PROMPT, INCOME_STATEMENT_USER_TEMPLATE,
            self.income_statement_parser, ticker, raw_data,
        )
        try:
            return (prompt | self.llm | self.income_statement_parser).invoke({})
        except Exception as e:
            print(f"Error processing Income Statement: {e}")
            return IncomeStatementAnalysis(
                summary="Error analyzing income statement.",
                key_metrics=["Unable to extract key metrics."],
                revenue_analysis="Analysis unavailable due to processing error.",
                profitability_analysis="Analysis unavailable due to processing error.",
                red_flags=[],
                outlook="Analysis unavailable due to processing error.",
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
            )

    def analyze_cashflow(
        self, ticker: str, raw_data: Dict[str, Any]
    ) -> CashFlowAnalysis:
        """Analyze cash flow statement XBRL data from 10-K (and optionally 10-Q)."""
        prompt = self._financial_statement_prompt(
            CASH_FLOW_SYSTEM_PROMPT, CASH_FLOW_USER_TEMPLATE,
            self.cash_flow_parser, ticker, raw_data,
        )
        try:
            return (prompt | self.llm | self.cash_flow_parser).invoke({})
        except Exception as e:
            print(f"Error processing Cash Flow: {e}")
            return CashFlowAnalysis(
                summary="Error analyzing cash flow statement.",
                key_metrics=["Unable to extract key metrics."],
                operating_cash_flow_analysis="Analysis unavailable due to processing error.",
                free_cash_flow_analysis="Analysis unavailable due to processing error.",
                red_flags=[],
                outlook="Analysis unavailable due to processing error.",
                sentiment_score=0.0,
                sentiment_analysis="Analysis unavailable due to processing error.",
            )
