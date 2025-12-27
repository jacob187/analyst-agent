# Entry point for main agent
import sys
from agents.sec_workflow.main_sec_workflow import SECWorkflow
from agents.technical_workflow.main_technical_workflow import TechnicalAgent
from database.local_logger import LocalLogger


class MainAgent:
    def __init__(self, ticker: str):
        self.sec_agent = SECWorkflow(ticker)
        self.technical_agent = TechnicalAgent(ticker)
        self.ticker = ticker
        self.logger = LocalLogger()

    def process_and_save(self) -> None:
        """Orchestrates processing and saving data from all sub-agents."""
        self.logger.log_message("INFO", f"Starting data processing for ticker: {self.ticker}")

        technical_success = self.technical_agent.process_and_save()
        if technical_success:
            self.logger.log_message("INFO", "Technical data processing completed successfully.")
        else:
            self.logger.log_message("ERROR", "Technical data processing failed. See logs for details.")

        sec_success = self.sec_agent.process_and_save()
        if sec_success:
            self.logger.log_message("INFO", "SEC data processing completed successfully.")
        else:
            self.logger.log_message("ERROR", "SEC data processing failed. See logs for details.")

        if technical_success and sec_success:
            self.logger.log_message("INFO", f"All data processing for {self.ticker} completed.")
        else:
            self.logger.log_message("WARNING", f"Data processing for {self.ticker} completed with some failures.")


if __name__ == "__main__":
    # Initialize a logger for the main script
    main_logger = LocalLogger()
    main_logger.log_message("INFO", "Starting Main Agent script.")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    main_agent = MainAgent(ticker)
    main_agent.process_and_save()
    main_logger.log_message("INFO", "Main Agent script finished.")
