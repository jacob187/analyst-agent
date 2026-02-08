"""
LangGraph workflow for SEC financial analysis with query planning.

Architecture:
- Simple queries → ReAct agent directly
- Complex queries → Planner → Step Executor (loop) → Synthesizer
"""

from typing import Dict, Any, TypedDict, Annotated, Optional, List, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from agents.prompts import (
    SEC_AGENT_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    STEP_EXECUTOR_PROMPT,
)
from agents.planner import (
    QueryPlanner,
    QueryPlan,
    QueryClassification,
    AnalysisStep,
    create_planner,
)


class AnalysisState(TypedDict):
    """Unified state for the analysis workflow."""

    # Core
    messages: Annotated[List[BaseMessage], add_messages]
    ticker: str

    # Planning
    query_complexity: str  # simple | moderate | complex
    classification: Optional[QueryClassification]
    plan: Optional[QueryPlan]

    # Execution
    current_step_index: int
    step_results: Dict[int, str]

    # Output
    final_response: str


def _get_latest_query(messages: List[BaseMessage]) -> str:
    """Extract the latest user query from messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _create_tools(ticker: str, llm: BaseChatModel, tavily_api_key: Optional[str] = None):
    """Create all available tools for the ticker."""
    from agents.tools.sec_tools import create_sec_tools

    tools, _ = create_sec_tools(ticker, llm)

    if tavily_api_key:
        from agents.tools.research_tools import create_research_tools

        research_tools = create_research_tools(ticker, tavily_api_key)
        tools.extend(research_tools)

    return tools


def _build_tools_dict(tools: List[Any]) -> Dict[str, Any]:
    """Convert tools list to name->func dict for direct execution."""
    return {tool.name: tool.func for tool in tools}


# =============================================================================
# Node Functions
# =============================================================================


def create_router_node(planner: QueryPlanner):
    """Create router node that classifies query complexity."""

    def router(state: AnalysisState) -> AnalysisState:
        query = _get_latest_query(state["messages"])

        if not query:
            state["query_complexity"] = "simple"
            state["classification"] = None
            return state

        # Classify the query
        classification = planner.classify_query(query)
        state["classification"] = classification

        # Determine complexity routing
        if classification.complexity == "simple" and classification.estimated_tools <= 1:
            state["query_complexity"] = "simple"
        else:
            state["query_complexity"] = "complex"

        return state

    return router


def create_react_node(react_agent: Any):
    """Create node that runs the ReAct agent for simple queries."""

    def react_node(state: AnalysisState) -> AnalysisState:
        result = react_agent.invoke({"messages": state["messages"]})

        if result and "messages" in result:
            response = result["messages"][-1].content
        else:
            response = str(result)

        state["final_response"] = response
        return state

    return react_node


def create_planner_node(planner: QueryPlanner):
    """Create node that generates an execution plan for complex queries."""

    def planner_node(state: AnalysisState) -> AnalysisState:
        query = _get_latest_query(state["messages"])

        plan = planner.create_plan(query)
        state["plan"] = plan
        state["current_step_index"] = 0
        state["step_results"] = {}

        return state

    return planner_node


def create_step_executor_node(tools_dict: Dict[str, Any], ticker: str):
    """Create node that executes the current plan step."""

    def step_executor(state: AnalysisState) -> AnalysisState:
        plan = state["plan"]
        if not plan or not plan.steps:
            return state

        step_index = state["current_step_index"]
        if step_index >= len(plan.steps):
            return state

        step = plan.steps[step_index]

        # Execute the tool directly
        if step.tool in tools_dict:
            try:
                result = tools_dict[step.tool]("")
                state["step_results"][step.id] = str(result)
            except Exception as e:
                state["step_results"][step.id] = f"[ERROR: {str(e)}]"
        else:
            state["step_results"][step.id] = f"[ERROR: Tool '{step.tool}' not found]"

        # Move to next step
        state["current_step_index"] = step_index + 1

        return state

    return step_executor


def create_synthesizer_node(llm: BaseChatModel, ticker: str):
    """Create node that synthesizes all step results into final response."""

    def synthesizer(state: AnalysisState) -> AnalysisState:
        plan = state["plan"]
        step_results = state["step_results"]
        query = _get_latest_query(state["messages"])

        # Format step results
        results_text = []
        if plan:
            for step in plan.steps:
                result = step_results.get(step.id, "[No result]")
                results_text.append(
                    f"=== Step {step.id}: {step.action} (via {step.tool}) ===\n{result}\n"
                )

        synthesis_approach = plan.synthesis_approach if plan else "Combine findings into a comprehensive answer"

        prompt = SYNTHESIS_SYSTEM_PROMPT.format(
            ticker=ticker,
            step_results="\n".join(results_text),
            query=query,
            synthesis_approach=synthesis_approach,
        )

        response = llm.invoke(prompt)
        state["final_response"] = response.content

        return state

    return synthesizer


# =============================================================================
# Routing Functions
# =============================================================================


def route_by_complexity(state: AnalysisState) -> Literal["react_agent", "planner"]:
    """Route based on query complexity."""
    if state["query_complexity"] == "simple":
        return "react_agent"
    return "planner"


def check_more_steps(state: AnalysisState) -> Literal["step_executor", "synthesizer"]:
    """Check if there are more steps to execute."""
    plan = state["plan"]
    if not plan or not plan.steps:
        return "synthesizer"

    if state["current_step_index"] < len(plan.steps):
        return "step_executor"

    return "synthesizer"


# =============================================================================
# Graph Construction
# =============================================================================


def create_planning_workflow(
    llm: BaseChatModel,
    ticker: str,
    tavily_api_key: Optional[str] = None,
) -> StateGraph:
    """
    Create the unified LangGraph workflow with planning capabilities.

    Flow:
    1. Router classifies query complexity
    2. Simple → ReAct agent → END
    3. Complex → Planner → Step Executor (loop) → Synthesizer → END
    """
    # Create tools and planner
    tools = _create_tools(ticker, llm, tavily_api_key)
    tools_dict = _build_tools_dict(tools)
    has_research = tavily_api_key is not None

    planner = create_planner(llm, ticker, has_research)

    # Create ReAct agent for simple queries
    system_prompt = SEC_AGENT_SYSTEM_PROMPT.format(ticker=ticker)
    react_agent = create_react_agent(llm, tools, prompt=system_prompt)

    # Build the graph
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("router", create_router_node(planner))
    workflow.add_node("react_agent", create_react_node(react_agent))
    workflow.add_node("planner", create_planner_node(planner))
    workflow.add_node("step_executor", create_step_executor_node(tools_dict, ticker))
    workflow.add_node("synthesizer", create_synthesizer_node(llm, ticker))

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_conditional_edges("router", route_by_complexity)
    workflow.add_edge("react_agent", END)
    workflow.add_edge("planner", "step_executor")
    workflow.add_conditional_edges("step_executor", check_more_steps)
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# =============================================================================
# Public API
# =============================================================================


class PlanningAgent:
    """
    Wrapper that provides a consistent interface for the planning workflow.

    Matches the interface expected by the API (invoke with messages dict).
    """

    def __init__(
        self,
        workflow: StateGraph,
        ticker: str,
    ):
        self.workflow = workflow
        self.ticker = ticker

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query through the planning workflow.

        Args:
            inputs: Dict with "messages" key containing conversation history

        Returns:
            Dict with "messages" key containing updated conversation
        """
        messages = inputs.get("messages", [])

        # Initialize state
        initial_state: AnalysisState = {
            "messages": messages,
            "ticker": self.ticker,
            "query_complexity": "",
            "classification": None,
            "plan": None,
            "current_step_index": 0,
            "step_results": {},
            "final_response": "",
        }

        # Run workflow
        final_state = self.workflow.invoke(initial_state)

        # Return in expected format
        response = final_state.get("final_response", "")
        if response:
            return {"messages": messages + [AIMessage(content=response)]}

        return {"messages": messages}


def create_planning_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
) -> PlanningAgent:
    """
    Create a planning-enabled agent for financial analysis.

    This is the main entry point for creating an agent with:
    - Query complexity classification
    - Multi-step planning for complex queries
    - Direct ReAct execution for simple queries
    - Result synthesis for planned queries

    Args:
        ticker: Company ticker symbol
        llm: Language model for processing
        tavily_api_key: Optional Tavily API key for web research tools

    Returns:
        PlanningAgent instance with invoke() method
    """
    workflow = create_planning_workflow(llm, ticker, tavily_api_key)
    return PlanningAgent(workflow, ticker)


# =============================================================================
# Backward Compatibility
# =============================================================================


def create_sec_qa_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
) -> PlanningAgent:
    """
    Create a Q&A agent with planning capabilities.

    This maintains backward compatibility with existing code while
    providing the new planning functionality.

    Args:
        ticker: Company ticker symbol
        llm: Language model for processing
        tavily_api_key: Optional Tavily API key for web research tools

    Returns:
        PlanningAgent instance (drop-in replacement for old ReAct agent)
    """
    return create_planning_agent(ticker, llm, tavily_api_key)
