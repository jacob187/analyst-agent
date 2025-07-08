from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI


from database.local_logger import LocalLogger
from agents.sec_workflow.sec_llm_models import SECDocumentProcessor
from agents.sec_workflow.get_SEC_data import SECDataRetrieval


class SECWorkflow:
    """Main SEC agent to process and store SEC data analysis."""

    def __init__(self, ticker: str):
        """Initialize with company ticker."""
        load_dotenv()

        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "Google API key must be set as an environment variable (GOOGLE_API_KEY)."
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20", api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.ticker = ticker
        self.data_retriever = SECDataRetrieval(ticker)
        self.document_processor = SECDocumentProcessor(llm)
        self.logger = LocalLogger()

    def process_and_save(self) -> None:
        """Process SEC data and save results to the JSON file."""
        # Step 1: Get the SEC data
        mda_text = self.data_retriever.extract_management_discussion()
        risk_text = self.data_retriever.extract_risk_factors()

        # Use the new JSON-serializable method for balance sheets
        balance_sheets = self.data_retriever.extract_balance_sheet_as_json()
        tenk = balance_sheets["tenk"]
        tenq = balance_sheets["tenq"]

        # Step 2: Process the data
        mda_analysis = self.document_processor.analyze_mda(self.ticker, mda_text)
        risk_analysis = self.document_processor.analyze_risk_factors(
            self.ticker, risk_text
        )
        balance_sheet_analysis = self.document_processor.analyze_balance_sheet(
            self.ticker, tenk, tenq
        )

        # Step 3: Combine results
        combined_analysis = {
            "ticker": self.ticker,
            "analysis_date": datetime.now().isoformat(),
            "management_discussion": mda_analysis.model_dump(),
            "risk_factors": risk_analysis.model_dump(),
            "balance_sheet_analysis": balance_sheet_analysis.model_dump(),
            "balance_sheets": balance_sheets,  # Now contains properly structured JSON
        }

        # Step 4: Read existing data first
        try:
            existing_data = self.logger.read_json()
        except Exception as e:
            print(f"Error reading existing data: {e}, creating new data structure")
            existing_data = {}

        # Step 5: Update with new analysis data under the "sec_data" key
        existing_data[self.ticker] = existing_data.get(
            self.ticker, {}
        )  # Ensure ticker key exists
        existing_data[self.ticker]["sec_data"] = combined_analysis

        # Step 6: Write the updated data back to the JSON file
        self.logger.write_json(existing_data)

        print(f"SEC analysis for {self.ticker} completed and saved to file.")

    def get_analysis(self) -> Dict[str, Any]:
        """Retrieve the SEC analysis from the JSON file."""
        try:
            data = self.logger.read_json()
            # Retrieve data from the 'sec_data' key
            ticker_data = data.get(self.ticker, {}).get("sec_data", {})
            if not ticker_data:
                print(f"No SEC data found for ticker {self.ticker}")
            return ticker_data
        except Exception as e:
            print(f"Error retrieving analysis: {e}")
            return {}


def main():
    """Example usage of the SEC Agent."""
    import os
    import sys
    from dotenv import load_dotenv
    from pathlib import Path

    # Load environment variables
    load_dotenv()

    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    # Create and run agent
    agent = SECWorkflow(ticker)
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
    md_content += "**Key Points:**\n\n"
    for point in mda["key_points"]:
        md_content += f"- {point}\n"

    # Add financial highlights section
    if "financial_highlights" in mda and mda["financial_highlights"]:
        md_content += "\n**Financial Highlights:**\n\n"
        for highlight in mda["financial_highlights"]:
            md_content += f"- {highlight}\n"

    # Add future outlook section
    if "future_outlook" in mda and mda["future_outlook"]:
        md_content += f"\n**Future Outlook:** {mda['future_outlook']}\n"

    md_content += (
        f"\n**Sentiment:** {mda['sentiment_score']} - {mda['sentiment_analysis']}\n\n"
    )

    md_content += "## Risk Factor Analysis\n\n"
    risk = analysis["risk_factors"]
    md_content += f"**Summary:** {risk['summary']}\n\n"
    md_content += "**Key Risks:**\n\n"
    for point in risk["key_risks"]:
        md_content += f"- {point}\n"

    # Add risk categories section
    if "risk_categories" in risk and risk["risk_categories"]:
        md_content += "\n**Risk Categories:**\n\n"
        for category, risks in risk["risk_categories"].items():
            md_content += f"\n### {category}\n"
            for r in risks:
                md_content += f"- {r}\n"

    md_content += f"\n**Risk Severity:** {risk['sentiment_score']} - {risk['sentiment_analysis']}\n"

    md_content += "\n## Balance Sheet Analysis\n\n"
    balance_sheet = analysis["balance_sheet_analysis"]
    md_content += f"**Summary:** {balance_sheet['summary']}\n\n"
    md_content += "**Key Metrics:**\n\n"
    for point in balance_sheet["key_metrics"]:
        md_content += f"- {point}\n"

    # Add the new fields from the updated model
    md_content += f"\n**Liquidity Analysis:** {balance_sheet['liquidity_analysis']}\n\n"
    md_content += f"**Solvency Analysis:** {balance_sheet['solvency_analysis']}\n\n"
    md_content += f"**Growth Trends:** {balance_sheet['growth_trends']}\n\n"

    # Add red flags section
    md_content += "**Red Flags:**\n\n"
    for flag in balance_sheet["red_flags"]:
        md_content += f"- {flag}\n"

    # Include financial highlights if present
    if "financial_highlights" in balance_sheet:
        md_content += "\n**Financial Highlights:**\n\n"
        for highlight in balance_sheet["financial_highlights"]:
            md_content += f"- {highlight}\n"

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
