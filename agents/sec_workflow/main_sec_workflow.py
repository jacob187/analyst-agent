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
            model="gemini-3-flash-preview", api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.ticker = ticker
        self.data_retriever = SECDataRetrieval(ticker)
        self.document_processor = SECDocumentProcessor(llm)
        self.logger = LocalLogger()

    def process_and_save(self) -> bool:
        """Process SEC data and save results to the JSON file."""
        self.logger.log_message("INFO", f"Starting SEC analysis for {self.ticker}")
        try:
            # Step 1: Get the SEC data with form-specific methods

            # Get MD&A from both 10-K and 10-Q
            mda_10k_data = self.data_retriever.get_mda_raw("10-K")
            self.logger.log_message("INFO", f"Retrieved 10-K MD&A data for {self.ticker}.")
            mda_10q_data = self.data_retriever.get_mda_raw("10-Q")
            self.logger.log_message("INFO", f"Retrieved 10-Q MD&A data for {self.ticker}.")

            # Get Risk Factors from both 10-K and 10-Q
            risk_10k_data = self.data_retriever.get_risk_factors_raw("10-K")
            self.logger.log_message("INFO", f"Retrieved 10-K Risk Factors data for {self.ticker}.")
            risk_10q_data = self.data_retriever.get_risk_factors_raw("10-Q")
            self.logger.log_message("INFO", f"Retrieved 10-Q Risk Factors data for {self.ticker}.")

            # Use the new JSON-serializable method for balance sheets
            balance_sheets = self.data_retriever.extract_balance_sheet_as_json()
            self.logger.log_message("INFO", f"Retrieved Balance Sheet data for {self.ticker}.")
            tenk = balance_sheets["tenk"]
            tenq = balance_sheets["tenq"]

            # Step 2: Process the data with form-specific analysis
            mda_10k_analysis = self.document_processor.analyze_mda(
                self.ticker, mda_10k_data
            )
            self.logger.log_message("INFO", f"Analyzed 10-K MD&A for {self.ticker}.")
            mda_10q_analysis = self.document_processor.analyze_mda(
                self.ticker, mda_10q_data
            )
            self.logger.log_message("INFO", f"Analyzed 10-Q MD&A for {self.ticker}.")

            risk_10k_analysis = self.document_processor.analyze_risk_factors(
                self.ticker, risk_10k_data
            )
            self.logger.log_message("INFO", f"Analyzed 10-K Risk Factors for {self.ticker}.")
            risk_10q_analysis = self.document_processor.analyze_risk_factors(
                self.ticker, risk_10q_data
            )
            self.logger.log_message("INFO", f"Analyzed 10-Q Risk Factors for {self.ticker}.")

            balance_sheet_analysis = self.document_processor.analyze_balance_sheet(
                self.ticker, tenk, tenq
            )
            self.logger.log_message("INFO", f"Analyzed Balance Sheet for {self.ticker}.")

            # Step 3: Combine results with form-specific sections
            combined_analysis = {
                "ticker": self.ticker,
                "analysis_date": datetime.now().isoformat(),
                "management_discussion_10k": mda_10k_analysis.model_dump(),
                "management_discussion_10q": mda_10q_analysis.model_dump(),
                "risk_factors_10k": risk_10k_analysis.model_dump(),
                "risk_factors_10q": risk_10q_analysis.model_dump(),
                "balance_sheet_analysis": balance_sheet_analysis.model_dump(),
                "balance_sheets": balance_sheets,  # Now contains properly structured JSON with metadata
                "raw_sections": {
                    "mda_10k": mda_10k_data,
                    "mda_10q": mda_10q_data,
                    "risk_10k": risk_10k_data,
                    "risk_10q": risk_10q_data,
                },
            }

            # Step 4: Read existing data first
            try:
                existing_data = self.logger.read_json()
                self.logger.log_message("INFO", "Existing data file read successfully.")
            except Exception as e:
                self.logger.log_message("ERROR", f"Error reading existing data: {e}, creating new data structure")
                existing_data = {}

            # Step 5: Update with new analysis data under the "sec_data" key
            existing_data[self.ticker] = existing_data.get(
                self.ticker, {}
            )  # Ensure ticker key exists
            existing_data[self.ticker]["sec_data"] = combined_analysis

            # Step 6: Write the updated data back to the JSON file
            self.logger.write_json(existing_data)
            self.logger.log_message("INFO", f"SEC analysis for {self.ticker} completed and saved to file.")
            return True
        except Exception as e:
            self.logger.log_message("ERROR", f"Error during SEC analysis orchestration for {self.ticker}: {e}")
            return False

    def get_analysis(self) -> Dict[str, Any]:
        """Retrieve the SEC analysis from the JSON file."""
        try:
            data = self.logger.read_json()
            # Retrieve data from the 'sec_data' key
            ticker_data = data.get(self.ticker, {}).get("sec_data", {})
            if not ticker_data:
                self.logger.log_message("WARNING", f"No SEC data found for ticker {self.ticker}")
            return ticker_data
        except Exception as e:
            self.logger.log_message("ERROR", f"Error retrieving analysis for {self.ticker}: {e}")
            return {}


def main():
    """Example usage of the SEC Agent."""
    import os
    import sys
    from dotenv import load_dotenv
    from pathlib import Path

    # Load environment variables
    load_dotenv()

    # Initialize a logger for the main script in this module
    main_logger = LocalLogger()
    main_logger.log_message("INFO", "Starting SECWorkflow main script.")

    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    # Create and run agent
    agent = SECWorkflow(ticker)
    success = agent.process_and_save()

    if not success:
        main_logger.log_message("ERROR", f"SEC analysis for {ticker} could not be completed due to errors. No report will be generated.")
        return

    # Generate a markdown report
    analysis = agent.get_analysis()
    if not analysis:
        main_logger.log_message("WARNING", "No analysis data found for ticker. Exiting.")
        return

    # Create markdown report
    md_content = f"# SEC Analysis for {ticker}\n\n"
    md_content += f"**Analysis Date:** {analysis['analysis_date']}\n\n"

    # Management Discussion Analysis - separate 10-K and 10-Q
    md_content += "## Management Discussion Analysis\n\n"

    # 10-K MD&A Analysis
    if "management_discussion_10k" in analysis:
        mda_10k = analysis["management_discussion_10k"]
        filing_info = mda_10k.get("filing_metadata", {})
        form = filing_info.get("form", "10-K")
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        md_content += f"### {form} Analysis\n\n"
        md_content += (
            f"**Filing Info:** {form} filed {filing_date}, Period: {period}\n\n"
        )
        md_content += f"**Summary:** {mda_10k['summary']}\n\n"
        md_content += "**Key Points:**\n\n"
        for point in mda_10k["key_points"]:
            md_content += f"- {point}\n"

        # Add financial highlights section
        if "financial_highlights" in mda_10k and mda_10k["financial_highlights"]:
            md_content += "\n**Financial Highlights:**\n\n"
            for highlight in mda_10k["financial_highlights"]:
                md_content += f"- {highlight}\n"

        # Add future outlook section
        if "future_outlook" in mda_10k and mda_10k["future_outlook"]:
            md_content += f"\n**Future Outlook:** {mda_10k['future_outlook']}\n"

        md_content += f"\n**Sentiment:** {mda_10k['sentiment_score']} - {mda_10k['sentiment_analysis']}\n\n"

    # 10-Q MD&A Analysis
    if "management_discussion_10q" in analysis:
        mda_10q = analysis["management_discussion_10q"]
        filing_info = mda_10q.get("filing_metadata", {})
        form = filing_info.get("form", "10-Q")
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        md_content += f"### {form} Analysis\n\n"
        md_content += (
            f"**Filing Info:** {form} filed {filing_date}, Period: {period}\n\n"
        )
        md_content += f"**Summary:** {mda_10q['summary']}\n\n"

        if mda_10q.get("key_points"):
            md_content += "**Key Points:**\n\n"
            for point in mda_10q["key_points"]:
                md_content += f"- {point}\n"

        # Add financial highlights section
        if "financial_highlights" in mda_10q and mda_10q["financial_highlights"]:
            md_content += "\n**Financial Highlights:**\n\n"
            for highlight in mda_10q["financial_highlights"]:
                md_content += f"- {highlight}\n"

        # Add future outlook section
        if "future_outlook" in mda_10q and mda_10q["future_outlook"]:
            md_content += f"\n**Future Outlook:** {mda_10q['future_outlook']}\n"

        md_content += f"\n**Sentiment:** {mda_10q['sentiment_score']} - {mda_10q['sentiment_analysis']}\n\n"

    # Risk Factor Analysis - separate 10-K and 10-Q
    md_content += "## Risk Factor Analysis\n\n"

    # 10-K Risk Analysis
    if "risk_factors_10k" in analysis:
        risk_10k = analysis["risk_factors_10k"]
        filing_info = risk_10k.get("filing_metadata", {})
        form = filing_info.get("form", "10-K")
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        md_content += f"### {form} Risk Analysis\n\n"
        md_content += (
            f"**Filing Info:** {form} filed {filing_date}, Period: {period}\n\n"
        )
        md_content += f"**Summary:** {risk_10k['summary']}\n\n"
        md_content += "**Key Risks:**\n\n"
        for point in risk_10k["key_risks"]:
            md_content += f"- {point}\n"

        # Add risk categories section
        if "risk_categories" in risk_10k and risk_10k["risk_categories"]:
            md_content += "\n**Risk Categories:**\n\n"
            for category, risks in risk_10k["risk_categories"].items():
                md_content += f"\n#### {category}\n"
                for r in risks:
                    md_content += f"- {r}\n"

        md_content += f"\n**Risk Severity:** {risk_10k['sentiment_score']} - {risk_10k['sentiment_analysis']}\n\n"

    # 10-Q Risk Analysis
    if "risk_factors_10q" in analysis:
        risk_10q = analysis["risk_factors_10q"]
        filing_info = risk_10q.get("filing_metadata", {})
        form = filing_info.get("form", "10-Q")
        filing_date = filing_info.get("filing_date", "Unknown")
        period = filing_info.get("period_of_report", "Unknown")

        md_content += f"### {form} Risk Analysis\n\n"
        md_content += (
            f"**Filing Info:** {form} filed {filing_date}, Period: {period}\n\n"
        )
        md_content += f"**Summary:** {risk_10q['summary']}\n\n"

        if risk_10q.get("key_risks"):
            md_content += "**Key Risks (Changes since last 10-K):**\n\n"
            for point in risk_10q["key_risks"]:
                md_content += f"- {point}\n"

        # Add risk categories section
        if "risk_categories" in risk_10q and risk_10q["risk_categories"]:
            md_content += "\n**Risk Categories:**\n\n"
            for category, risks in risk_10q["risk_categories"].items():
                md_content += f"\n#### {category}\n"
                for r in risks:
                    md_content += f"- {r}\n"

        md_content += f"\n**Risk Severity:** {risk_10q['sentiment_score']} - {risk_10q['sentiment_analysis']}\n"

    md_content += "\n## Balance Sheet Analysis\n\n"
    balance_sheet = analysis["balance_sheet_analysis"]

    # Add filing information from balance sheet metadata
    balance_sheets_data = analysis.get("balance_sheets", {})
    if "tenk_metadata" in balance_sheets_data:
        tenk_meta = balance_sheets_data["tenk_metadata"]
        md_content += f"**10-K Filing Info:** Form {tenk_meta.get('form', '10-K')} filed {tenk_meta.get('filing_date', 'Unknown')}, Period: {tenk_meta.get('period_of_report', 'Unknown')}\n\n"
    if "tenq_metadata" in balance_sheets_data:
        tenq_meta = balance_sheets_data["tenq_metadata"]
        md_content += f"**10-Q Filing Info:** Form {tenq_meta.get('form', '10-Q')} filed {tenq_meta.get('filing_date', 'Unknown')}, Period: {tenq_meta.get('period_of_report', 'Unknown')}\n\n"

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

    main_logger.log_message("INFO", f"Analysis complete! Report saved to {report_path}")


if __name__ == "__main__":
    main()
