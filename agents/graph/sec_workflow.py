from typing import Dict, Any, TypedDict, Annotated
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


class SECState(TypedDict):
    """State for SEC analysis workflow using granular SEC tools."""

    ticker: str
    llm_id: str
    available_tools: list
    messages: Annotated[list, add_messages]
    current_step: str


def create_sec_agent(llm: BaseChatModel, ticker: str) -> Any:
    """
    Create a ReAct agent with granular SEC tools.
    The agent can choose which specific tool to use based on the query.
    """
    from agents.tools.sec_tools import create_sec_tools

    # Create tools using proper separation of concerns
    tools, llm_id = create_sec_tools(ticker, llm)

    # Create a ReAct agent with the tools
    agent = create_react_agent(llm, tools)

    return agent, llm_id


def create_sec_workflow(llm: BaseChatModel) -> StateGraph:
    """
    Create a LangGraph workflow that initializes SEC tools and makes them available.
    This ensures only one SEC API call is made.
    """

    def initialize_sec_tools(state: SECState) -> SECState:
        """Initialize SEC tools using proper separation of concerns."""
        from agents.tools.sec_tools import create_sec_tools

        ticker = state["ticker"]
        print(f"Initializing SEC tools factory for {ticker}")

        # This makes the single SEC API call
        tools, llm_id = create_sec_tools(ticker, llm)
        state["llm_id"] = llm_id  # Store the LLM ID for reference
        state["available_tools"] = tools
        state["current_step"] = "initialized"
        state["messages"].append(
            f"Initialized SEC tools for {ticker} with {len(state['available_tools'])} tools available"
        )
        return state

    # Create the workflow
    workflow = StateGraph(SECState)

    # Add nodes
    workflow.add_node("initialize", initialize_sec_tools)

    # Set entry point and edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", END)

    return workflow.compile()


def initialize_sec_tools_for_ticker(ticker: str, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Initialize SEC tools for a ticker and return the factory and available tools.

    Args:
        ticker: Company ticker symbol
        llm: Language model for processing

    Returns:
        Dictionary containing the tools factory and available tools
    """
    workflow = create_sec_workflow(llm)

    initial_state = SECState(
        ticker=ticker,
        llm_id="",
        available_tools=[],
        messages=[],
        current_step="starting",
    )

    # Run the workflow
    final_state = workflow.invoke(initial_state)

    return {
        "ticker": ticker,
        "status": final_state["current_step"],
        "llm_id": final_state["llm_id"],
        "available_tools": final_state["available_tools"],
        "messages": final_state["messages"],
    }


def create_sec_qa_agent(ticker: str, llm: BaseChatModel) -> Any:
    """
    Create a Q&A agent that can answer questions about SEC filings using granular tools.

    Args:
        ticker: Company ticker symbol
        llm: Language model for processing

    Returns:
        Configured agent that can answer SEC-related questions
    """
    agent, tools_factory = create_sec_agent(llm, ticker)
    return agent
