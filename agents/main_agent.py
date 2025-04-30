# Entry point for main agent
import sys
from agents.sec_agent.main_sec_agent import SECAgent
from agents.technical_agent.main_technical_agent import TechnicalAgent


class MainAgent:
    def __init__(self, ticker: str):
        self.sec_agent = SECAgent(ticker)
        self.technical_agent = TechnicalAgent(ticker)

    def process_and_save(self):
        self.technical_agent.process_and_save()
        self.sec_agent.process_and_save()


if __name__ == "__main__":
    print("Starting Main Agent")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    main_agent = MainAgent(ticker)
    main_agent.process_and_save()
