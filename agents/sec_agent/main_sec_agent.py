from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from agents.sec_agent.get_SEC_data import SECDataRetrieval
from database.local_logger import LocalLogger


# Pydantic models for structured output
class SentimentAnalysis(BaseModel):
    """Model for sentiment analysis of text."""

    score: float = Field(
        description="Sentiment score from -10 (very negative) to 10 (very positive)"
    )
    analysis: str = Field(description="Detailed explanation of the sentiment score")


class MDnAAnalysis(BaseModel):
    """Full analysis for Management Discussion and Analysis"""

    ticker: str = Field(description="Company ticker symbol")
    summary: str = Field(description="Concise summary of the MD&A section")
    key_points: List[str] = Field(description="List of key points from MD&A")
    financial_highlights: List[str] = Field(
        description="Key financial metrics and trends discussed"
    )
    future_outlook: str = Field(description="Management's outlook for the future")
    sentiment_score: float = Field(description="Overall sentiment score")
    sentiment_analysis: str = Field(description="Detailed sentiment explanation")
    comparison: Optional[str] = Field(
        None, description="Comparison between 10-K and 10-Q if both are available"
    )


class RiskFactorAnalysis(BaseModel):
    """Full analysis for Risk Factors section"""

    ticker: str = Field(description="Company ticker symbol")
    summary: str = Field(description="Concise summary of risk factors")
    key_risks: List[str] = Field(description="List of key risk factors")
    risk_categories: Dict[str, List[str]] = Field(
        description="Risk factors categorized by type"
    )
    sentiment_score: float = Field(description="Overall risk severity score")
    sentiment_analysis: str = Field(description="Detailed risk assessment explanation")
    comparison: Optional[str] = Field(
        None, description="Comparison between 10-K and 10-Q if both are available"
    )


class SECDocumentProcessor:
    """Processes SEC documents using LLM."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key must be provided")

        self.llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini-2024-07-18")

        # Initialize output parsers
        self.mda_parser = PydanticOutputParser(pydantic_object=MDnAAnalysis)
        self.risk_parser = PydanticOutputParser(pydantic_object=RiskFactorAnalysis)

    def generate_mda_prompt(self, ticker: str, mda_text: str) -> ChatPromptTemplate:
        """Generate a prompt for analyzing Management Discussion and Analysis from 10-K only."""
        system_message = """You are a financial expert analyzing the Management Discussion and Analysis (MD&A) 
        section of SEC filings. Provide a comprehensive analysis including a summary, key points, 
        financial highlights, future outlook, and sentiment analysis.
        
        IMPORTANT: You must respond with a properly formatted JSON object that matches the schema exactly.
        DO NOT return the schema definition - fill in actual values based on your analysis.
        """

        user_template = """Analyze the following MD&A section from {ticker}'s 10-K SEC filing:
        
        {mda_text}... [truncated for brevity]
        
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
            mda_text=mda_text[:3000],
            format_instructions=self.mda_parser.get_format_instructions(),
        )

    def generate_risk_factors_prompt(
        self, ticker: str, risk_text: str
    ) -> ChatPromptTemplate:
        """Generate a prompt for analyzing Risk Factors from 10-K only."""
        system_message = """You are a financial risk analyst examining the Risk Factors section of SEC filings.
        Provide a comprehensive analysis including a summary, key risks, risk categorization,
        and an overall assessment of risk severity.
        
        IMPORTANT: You must respond with a properly formatted JSON object that matches the schema exactly.
        DO NOT return the schema definition - fill in actual values based on your analysis.
        """

        user_template = """Analyze the following Risk Factors section from {ticker}'s 10-K SEC filing:
        
        {risk_text}... [truncated for brevity]
        
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
            risk_text=risk_text[:3000],
            format_instructions=self.risk_parser.get_format_instructions(),
        )

    def analyze_mda(self, ticker: str, mda_text: str) -> MDnAAnalysis:
        """Analyze the Management Discussion section from 10-K."""
        prompt = self.generate_mda_prompt(ticker, mda_text)
        try:
            # Maintain the Langchain chain approach
            chain = prompt | self.llm
            response = chain.invoke({})

            # Process the response content
            content = response.content

            # Try to parse as JSON directly
            import json

            try:
                # First attempt: direct JSON parsing
                data = json.loads(content)
                return MDnAAnalysis(**data)
            except json.JSONDecodeError:
                # Second attempt: extract JSON from markdown code blocks
                import re

                json_match = re.search(
                    r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    return MDnAAnalysis(**data)
                raise ValueError(f"Could not extract valid JSON from LLM response")
        except Exception as e:
            print(f"Error processing MD&A: {e}")
            # Return fallback values
            return MDnAAnalysis(
                ticker=ticker,
                summary="Error analyzing MD&A section.",
                key_points=["Unable to extract key points."],
                financial_highlights=["Unable to extract financial highlights."],
                future_outlook="Unable to determine future outlook.",
                sentiment_score=0.0,
                sentiment_analysis="Unable to determine sentiment.",
            )

    def analyze_risk_factors(self, ticker: str, risk_text: str) -> RiskFactorAnalysis:
        """Analyze the Risk Factors section from 10-K."""
        prompt = self.generate_risk_factors_prompt(ticker, risk_text)
        try:
            # Maintain the Langchain chain approach
            chain = prompt | self.llm
            response = chain.invoke({})

            # Process the response content
            content = response.content

            # Try to parse as JSON directly
            import json

            try:
                # First attempt: direct JSON parsing
                data = json.loads(content)
                return RiskFactorAnalysis(**data)
            except json.JSONDecodeError:
                # Second attempt: extract JSON from markdown code blocks
                import re

                json_match = re.search(
                    r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    return RiskFactorAnalysis(**data)
                raise ValueError(f"Could not extract valid JSON from LLM response")
        except Exception as e:
            print(f"Error processing risk factors: {e}")
            # Return fallback values
            return RiskFactorAnalysis(
                ticker=ticker,
                summary="Error analyzing risk factors.",
                key_risks=["Unable to extract key risks."],
                risk_categories={"General": ["Unable to categorize risks."]},
                sentiment_score=0.0,
                sentiment_analysis="Unable to determine risk severity.",
            )


class SECAgent:
    """Main SEC agent to process and store SEC data analysis."""

    def __init__(self, ticker: str):
        """Initialize with company ticker."""
        self.ticker = ticker
        self.data_retriever = SECDataRetrieval(ticker)
        self.document_processor = SECDocumentProcessor()
        self.logger = LocalLogger()

    def process_and_save(self) -> None:
        """Process SEC data and save results to the JSON file."""
        # Step 1: Get the SEC data
        mda_text = self.data_retriever.extract_management_discussion()
        risk_text = self.data_retriever.extract_risk_factors()

        # Use the new JSON-serializable method for balance sheets
        balance_sheets = self.data_retriever.extract_balance_sheet_as_json()

        # Step 2: Process the data
        mda_analysis = self.document_processor.analyze_mda(self.ticker, mda_text)
        risk_analysis = self.document_processor.analyze_risk_factors(
            self.ticker, risk_text
        )

        # Step 3: Combine results
        combined_analysis = {
            "ticker": self.ticker,
            "analysis_date": datetime.now().isoformat(),
            "management_discussion": mda_analysis.model_dump(),
            "risk_factors": risk_analysis.model_dump(),
            "balance_sheets": balance_sheets,  # Now contains properly structured JSON
        }

        # Step 4: Read existing data first
        try:
            existing_data = self.logger.read_json()
        except Exception as e:
            print(f"Error reading existing data: {e}, creating new data structure")
            existing_data = {}

        # Step 5: Update with new analysis data using ticker as key
        existing_data[self.ticker] = combined_analysis

        # Step 6: Write the updated data back to the JSON file
        self.logger.write_json(existing_data)

        print(f"Analysis for {self.ticker} completed and saved to file.")

    def get_analysis(self) -> Dict[str, Any]:
        """Retrieve the analysis from the JSON file."""
        try:
            data = self.logger.read_json()
            ticker_data = data.get(self.ticker, {})
            if not ticker_data:
                print(f"No data found for ticker {self.ticker}")
            return ticker_data
        except Exception as e:
            print(f"Error retrieving analysis: {e}")
            return {}


def main():
    """Example usage of the SEC Agent."""
    import os
    from dotenv import load_dotenv
    from pathlib import Path

    # Load environment variables
    load_dotenv()

    # Get ticker from command line or use default
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"Analyzing {ticker}...")
    agent = SECAgent(ticker)
    agent.process_and_save()

    # Generate a markdown report
    analysis = agent.get_analysis()
    if not analysis:
        print("No analysis data found for ticker.")
        return

    # Create markdown report
    md_content = f"# SEC Analysis for {ticker}\n\n"
    md_content += f"**Analysis Date:** {analysis['analysis_date']}\n\n"

    md_content += "## Management Discussion Analysis\n\n"
    mda = analysis["management_discussion"]
    md_content += f"**Summary:** {mda['summary']}\n\n"
    md_content += "**Key Points:**\n"
    for point in mda["key_points"]:
        md_content += f"- {point}\n"
    md_content += (
        f"\n**Sentiment:** {mda['sentiment_score']} - {mda['sentiment_analysis']}\n\n"
    )

    md_content += "## Risk Factor Analysis\n\n"
    risk = analysis["risk_factors"]
    md_content += f"**Summary:** {risk['summary']}\n\n"
    md_content += "**Key Risks:**\n"
    for point in risk["key_risks"]:
        md_content += f"- {point}\n"
    md_content += f"\n**Risk Severity:** {risk['sentiment_score']} - {risk['sentiment_analysis']}\n"

    # Get the project root directory for saving the file
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_dir = root_dir / "data"
    os.makedirs(data_dir, exist_ok=True)

    # Save markdown report in the data directory
    report_path = data_dir / f"{ticker}_analysis.md"
    with open(report_path, "w") as f:
        f.write(md_content)

    print(f"Analysis complete! Report saved to {report_path}")


if __name__ == "__main__":
    main()
