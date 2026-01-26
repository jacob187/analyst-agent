from typing import Dict, Any, TypedDict, Annotated, Optional
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


def create_sec_agent(
    llm: BaseChatModel,
    ticker: str,
    checkpointer=None,
    tavily_api_key: Optional[str] = None,
) -> Any:
    """
    Create a ReAct agent with granular SEC tools and optional research tools.
    The agent can choose which specific tool to use based on the query.

    Args:
        llm: Language model for the agent
        ticker: Company ticker symbol
        checkpointer: Optional checkpointer for conversation memory
        tavily_api_key: Optional Tavily API key for web research tools
    """
    from agents.tools.sec_tools import create_sec_tools

    # Create SEC tools using proper separation of concerns
    tools, llm_id = create_sec_tools(ticker, llm)

    # Add research tools if Tavily API key is provided
    if tavily_api_key:
        from agents.tools.research_tools import create_research_tools

        research_tools = create_research_tools(ticker, tavily_api_key)
        tools.extend(research_tools)

    # System prompt - tool descriptions come from the Tool objects themselves
    system_prompt = f"""You are a financial analyst assistant for {ticker}.

    You have access to tools for SEC filings, stock market data, and web research.
    ALWAYS use the appropriate tool to answer questions - do not rely on your own knowledge.

    Tool selection guidance:
    - For stock prices, P/E ratios (trailing/forward), market cap, valuation metrics → use stock market tools
    - For technical indicators (RSI, MACD, moving averages, Bollinger Bands) → use technical analysis tools
    - For risks, management outlook, financial statements from filings → use SEC filing tools
    - For current news, competitors, industry trends → use research tools (if available)

    If you're unsure which tool has the data, try the most likely tool rather than refusing.
    Report what data is available or unavailable based on the tool's response."""

    # Create a ReAct agent with the tools, system prompt, and optional checkpointer
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )

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


def create_sec_qa_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
) -> Any:
    """
    Create a Q&A agent that can answer questions about SEC filings using granular tools.

    Args:
        ticker: Company ticker symbol
        llm: Language model for processing
        tavily_api_key: Optional Tavily API key for web research tools

    Returns:
        Configured agent that can answer SEC-related questions
    """
    agent, tools_factory = create_sec_agent(
        llm, ticker, tavily_api_key=tavily_api_key
    )
    return agent
