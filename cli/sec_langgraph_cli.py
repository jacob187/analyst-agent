"""
SEC CLI Chatbot using LangGraph ReAct Architecture
Uses the existing sec_workflow.py with proper memory management
"""

import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph.sec_graph import create_sec_agent


class SECLangGraphCLI:
    """CLI interface using LangGraph ReAct agent with memory."""

    def __init__(self):
        self.agent = None
        self.llm_id = None
        self.config = None
        self.current_ticker = None

    def initialize(self) -> bool:
        """Initialize the LangGraph agent."""
        load_dotenv()

        if "GOOGLE_API_KEY" not in os.environ:
            print("❌ Error: GOOGLE_API_KEY not found in environment variables")
            print("Please set your Google API key in .env file")
            return False

        try:
            print("🔄 Initializing SEC LangGraph Agent...")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.environ["GOOGLE_API_KEY"],
            )

            # Get user input for ticker
            ticker = self._get_initial_ticker()
            if not ticker:
                return False

            # Create agent with tools for the ticker
            self.agent, self.llm_id = create_sec_agent(llm, ticker)
            self.current_ticker = ticker

            # Set up memory configuration
            memory = MemorySaver()
            self.config = {
                "configurable": {"thread_id": "sec_chat_session"},
                "checkpointer": memory,
            }

            print(f"✅ SEC Agent ready for {ticker}!")
            print(f"📊 Available tools: 8 SEC analysis tools")

            return True

        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False

    def _get_initial_ticker(self) -> Optional[str]:
        """Get the initial ticker from user."""
        print("\n🎯 Which company would you like to analyze?")
        print("Enter a ticker symbol (e.g., AAPL, MSFT, GOOGL)")

        while True:
            ticker = input("📈 Ticker: ").strip().upper()

            if not ticker:
                print("Please enter a ticker symbol or 'quit' to exit")
                continue

            if ticker.lower() in ["quit", "exit", "q"]:
                return None

            if len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha():
                return ticker
            else:
                print("Please enter a valid ticker symbol (1-5 letters)")

    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 60)
        print("🤖 SEC LANGGRAPH CHAT ASSISTANT")
        print("=" * 60)
        print(f"🎯 Analyzing: {self.current_ticker}")
        print("\nI can help you with:")
        print("• Risk analysis and specific risk factors")
        print("• Management discussion and outlook")
        print("• Balance sheet analysis and financial health")
        print("• Raw data searches for specific phrases")
        print("• Comprehensive company summaries")
        print("\nJust ask me naturally! Examples:")
        print("• 'What are the main risks?'")
        print("• 'Tell me about management outlook'")
        print("• 'Find mentions of cybersecurity in risk factors'")
        print("• 'How is the financial health?'")
        print("\nCommands:")
        print("• 'help' - Show help")
        print("• 'switch [TICKER]' - Switch to different company")
        print("• 'tools' - List available analysis tools")
        print("• 'memory' - Show conversation memory")
        print("• 'quit' - Exit")
        print("=" * 60)

    def print_help(self):
        """Print detailed help."""
        print("\n📚 HELP - SEC LangGraph Assistant")
        print("-" * 50)
        print("🎯 How it works:")
        print("  • I'm a ReAct agent that chooses the right tool for your question")
        print("  • I have access to both raw SEC data and AI-processed summaries")
        print("  • I remember our conversation context automatically")
        print("\n💡 Question types I handle well:")
        print("  • 'What are the risks?' → Uses risk summary tool")
        print("  • 'Find specific phrase in risks' → Uses raw risk data tool")
        print("  • 'How is financial health?' → Uses balance sheet summary")
        print("  • 'What's the exact debt amount?' → Uses raw balance sheet")
        print("  • 'Give me an overview' → Uses comprehensive summary")
        print("\n🔧 Commands:")
        print("  • switch AAPL - Switch to analyzing Apple")
        print("  • tools - See what analysis tools I have")
        print("  • memory - View our conversation history")
        print("  • help - Show this help")
        print("  • quit - Exit the assistant")
        print("-" * 50)

    def show_tools(self):
        """Show available tools."""
        print(f"\n🛠️ Available Analysis Tools:")
        print("-" * 50)

        tools_info = [
            (
                "Risk Factors Summary",
                "Get structured analysis of company risks with sentiment scoring",
            ),
            (
                "Raw Risk Factors",
                "Get complete raw text of risk factors for phrase searching",
            ),
            (
                "MD&A Summary",
                "Get management discussion analysis with key points and outlook",
            ),
            (
                "Raw Management Discussion",
                "Get complete raw text of management discussion",
            ),
            (
                "Balance Sheet Summary",
                "Get financial health analysis with key metrics and red flags",
            ),
            (
                "Raw Balance Sheets",
                "Get complete raw balance sheet data for detailed analysis",
            ),
            (
                "Complete 10-K Text",
                "Get all major sections of the 10-K filing as raw text",
            ),
            ("All Summaries", "Get comprehensive overview of all sections at once"),
        ]

        for name, description in tools_info:
            print(f"• {name}: {description}")

        print("-" * 50)
        print("💡 I automatically choose the right tool based on your question!")
        print("📝 Examples:")
        print("  • 'What are the risks?' → Uses Risk Factors Summary")
        print("  • 'Find cybersecurity in risks' → Uses Raw Risk Factors")
        print("  • 'How is financial health?' → Uses Balance Sheet Summary")

    def show_memory(self):
        """Show conversation memory."""
        try:
            # Get the conversation state from the agent
            print("\n🧠 Conversation Memory:")
            print("-" * 30)
            print("(LangGraph manages conversation memory automatically)")
            print("I remember:")
            print(f"• Current company: {self.current_ticker}")
            print("• Previous questions and answers in this session")
            print("• Context from our ongoing conversation")
            print("-" * 30)
        except Exception as e:
            print(f"❌ Could not retrieve memory: {e}")

    def switch_ticker(self, new_ticker: str) -> bool:
        """Switch to a different ticker."""
        if not new_ticker or not new_ticker.isalpha():
            print("❌ Please provide a valid ticker symbol")
            return False

        try:
            print(f"🔄 Switching to {new_ticker.upper()}...")

            # Get LLM from current agent
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.environ["GOOGLE_API_KEY"],
            )

            # Create new agent for new ticker
            self.agent, self.llm_id = create_sec_agent(llm, new_ticker.upper())
            self.current_ticker = new_ticker.upper()

            # Reset memory for new company
            memory = MemorySaver()
            self.config = {
                "configurable": {"thread_id": f"sec_chat_{new_ticker.lower()}"},
                "checkpointer": memory,
            }

            print(f"✅ Now analyzing {self.current_ticker}")
            return True

        except Exception as e:
            print(f"❌ Failed to switch to {new_ticker}: {e}")
            return False

    def process_user_input(self, user_input: str) -> bool:
        """Process user input and return False if should exit."""
        user_input = user_input.strip()

        # Handle commands
        if user_input.lower() in ["quit", "exit", "q"]:
            return False
        elif user_input.lower() == "help":
            self.print_help()
            return True
        elif user_input.lower() == "tools":
            self.show_tools()
            return True
        elif user_input.lower() == "memory":
            self.show_memory()
            return True
        elif user_input.lower().startswith("switch "):
            new_ticker = user_input[7:].strip()
            self.switch_ticker(new_ticker)
            return True
        elif not user_input:
            return True

        # Process with LangGraph agent
        try:
            print("\n🤖 Thinking...")

            # Create message for the agent, always include current ticker as context
            # so the agent does not ask for the company again between turns.
            messages = [
                SystemMessage(
                    content=(
                        f"Current company ticker: {self.current_ticker}. "
                        "Use the SEC tools to answer questions for this company. "
                        "Do not ask for the company unless the user requests a switch."
                    )
                ),
                HumanMessage(content=user_input),
            ]

            # Invoke the agent with memory
            result = self.agent.invoke({"messages": messages}, config=self.config)

            # Extract and display the response
            if result and "messages" in result:
                # Get the last AI message
                ai_messages = [
                    msg for msg in result["messages"] if isinstance(msg, AIMessage)
                ]
                if ai_messages:
                    response = ai_messages[-1].content
                    print(f"\n🤖 Assistant:")
                    print("-" * 40)
                    print(response)
                else:
                    print(
                        "\n🤖 I processed your request but didn't generate a response."
                    )
            else:
                print("\n❌ No response generated")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")

        return True

    def run(self):
        """Run the CLI interface."""
        if not self.initialize():
            return

        self.print_welcome()

        try:
            while True:
                # Show current context
                context = f" [{self.current_ticker}]"

                # Get user input
                user_input = input(f"\n💬 You{context}: ").strip()

                # Process input
                if not self.process_user_input(user_input):
                    break

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except EOFError:
            print("\n\n👋 Goodbye!")


def main():
    """Main entry point."""
    cli = SECLangGraphCLI()
    cli.run()


if __name__ == "__main__":
    main()
