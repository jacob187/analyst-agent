"""
LangGraph workflow for financial analysis with query planning.

Architecture:
- Simple queries → ReAct agent directly
- Complex queries → Planner → Step Executor (loop) → Synthesizer

Streaming:
- stream_sync() yields events as the graph executes (node transitions,
  tool calls, thinking blocks, tokens, final response)
- invoke() remains unchanged for backward compatibility
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, TypedDict, Annotated, Optional, List, Literal, Generator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from agents.llm_utils import extract_text, parse_llm_response
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


def _create_tools(ticker: str, llm: BaseChatModel, tavily_api_key: Optional[str] = None, sec_header: str = "", user_id: str | None = None):
    """Create all available tools for the ticker."""
    from agents.tools.sec_tools import create_sec_tools
    from agents.tools.stock_tools import create_stock_tools
    from agents.tools.market_tools import create_market_tools

    tools, _ = create_sec_tools(ticker, llm, sec_header)
    tools.extend(create_stock_tools(ticker))
    tools.extend(create_market_tools())

    if tavily_api_key:
        from agents.tools.research_tools import create_research_tools

        research_tools = create_research_tools(ticker, tavily_api_key)
        tools.extend(research_tools)

    # Briefing history tools (always available — returns "no history" if DB is empty)
    from agents.tools.briefing_tools import create_briefing_tools
    tools.extend(create_briefing_tools(ticker, user_id=user_id))

    return tools


def _build_tools_dict(tools: List[Any]) -> Dict[str, Any]:
    """Convert tools list to name->func dict for direct execution."""
    return {tool.name: tool.func for tool in tools}


# =============================================================================
# Node Functions
# =============================================================================


UNCLEAR_QUERY_RESPONSE = (
    "I couldn't understand that message. Could you rephrase your question? "
    "For example, you can ask about stock price, financials, risk factors, "
    "technical indicators, or recent SEC filings."
)


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

        # Unclear queries get a clarification response — no tools invoked
        if classification.complexity == "unclear":
            state["query_complexity"] = "unclear"
            state["final_response"] = UNCLEAR_QUERY_RESPONSE
            return state

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
            # extract_text handles the case where Gemini 3 returns
            # content as a list of blocks instead of a plain string
            response = extract_text(result["messages"][-1].content)
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


def _build_dependency_layers(steps: List[AnalysisStep]) -> List[List[AnalysisStep]]:
    """Group steps into layers where each layer's dependencies are satisfied by prior layers.

    Steps declare which earlier step IDs they depend on via `depends_on`.
    Layer 0 contains all steps with no dependencies — these run first, in parallel.
    Layer 1 contains steps whose dependencies are all in layer 0, and so on.

    This is a topological sort by depth. Steps within the same layer are independent
    of each other and safe to execute concurrently.
    """
    completed_ids: set[int] = set()
    remaining = list(steps)
    layers: List[List[AnalysisStep]] = []

    while remaining:
        # Find steps whose dependencies are all satisfied
        ready = [s for s in remaining if all(d in completed_ids for d in s.depends_on)]

        if not ready:
            # Circular dependency or missing IDs — flush remaining to avoid infinite loop
            layers.append(remaining)
            break

        layers.append(ready)
        completed_ids.update(s.id for s in ready)
        remaining = [s for s in remaining if s.id not in completed_ids]

    return layers


def _execute_step(step: AnalysisStep, tools_dict: Dict[str, Any]) -> tuple[int, str]:
    """Execute a single step and return (step_id, result). Thread-safe helper."""
    if step.tool not in tools_dict:
        return step.id, f"[ERROR: Tool '{step.tool}' not found]"
    try:
        return step.id, str(tools_dict[step.tool](""))
    except Exception as e:
        return step.id, f"[ERROR: {str(e)}]"


def create_step_executor_node(tools_dict: Dict[str, Any], ticker: str):
    """Create node that executes all plan steps, running independent steps in parallel.

    Steps are grouped into dependency layers via _build_dependency_layers().
    Each layer runs concurrently with ThreadPoolExecutor — only steps that share
    a layer run at the same time, and a layer only starts once the previous layer
    has fully completed. This respects the depends_on declarations from the planner
    while maximising throughput.
    """

    def step_executor(state: AnalysisState) -> AnalysisState:
        plan = state["plan"]
        if not plan or not plan.steps:
            return state

        layers = _build_dependency_layers(plan.steps)
        step_results: Dict[int, str] = {}

        for layer in layers:
            if len(layer) == 1:
                # Single step — no thread overhead needed
                step_id, result = _execute_step(layer[0], tools_dict)
                step_results[step_id] = result
            else:
                # Multiple independent steps — run concurrently
                with ThreadPoolExecutor(max_workers=len(layer)) as pool:
                    futures = {
                        pool.submit(_execute_step, step, tools_dict): step
                        for step in layer
                    }
                    for future in as_completed(futures):
                        step_id, result = future.result()
                        step_results[step_id] = result

        state["step_results"] = step_results
        state["current_step_index"] = len(plan.steps)
        return state

    return step_executor


def _build_synthesis_prompt(state: AnalysisState, ticker: str) -> str:
    """Build the synthesis prompt from plan state. Shared by streaming and non-streaming paths."""
    plan = state["plan"]
    step_results = state["step_results"]
    query = _get_latest_query(state["messages"])

    results_text = []
    if plan:
        for step in plan.steps:
            result = step_results.get(step.id, "[No result]")
            results_text.append(
                f"=== Step {step.id}: {step.action} (via {step.tool}) ===\n{result}\n"
            )

    synthesis_approach = plan.synthesis_approach if plan else "Combine findings into a comprehensive answer"

    return SYNTHESIS_SYSTEM_PROMPT.format(
        ticker=ticker,
        step_results="\n".join(results_text),
        query=query,
        synthesis_approach=synthesis_approach,
    )


def _process_streaming_chunk(chunk, writer) -> str:
    """Process a single LLM streaming chunk, emitting events via writer.

    Uses parse_llm_response from llm_utils for consistent parsing,
    then emits thinking/token events via the stream writer.

    Returns the text content extracted from this chunk (for accumulation).
    """
    parsed = parse_llm_response(chunk)

    if parsed.thinking:
        writer({"type": "thinking", "message": parsed.thinking})
    if parsed.text:
        writer({"type": "token", "message": parsed.text})

    return parsed.text


def create_synthesizer_node(llm: BaseChatModel, ticker: str):
    """Create node that synthesizes all step results into final response.

    Supports two execution modes:
    - Streaming (via workflow.stream): uses get_stream_writer() to emit
      thinking blocks and tokens in real-time as the LLM generates them.
    - Non-streaming (via workflow.invoke): falls back to llm.invoke() for
      the complete response at once. This keeps backward compatibility.
    """

    async def synthesizer(state: AnalysisState) -> AnalysisState:
        prompt = _build_synthesis_prompt(state, ticker)

        # Try to get a stream writer — only available inside workflow.stream()
        writer = None
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            pass

        if writer:
            # Streaming path: emit tokens/thinking as they arrive.
            # astream() is the async equivalent — avoids blocking the event loop
            # for the duration of synthesis (which can be 10-30s with thinking enabled).
            full_response = ""
            async for chunk in llm.astream(prompt):
                full_response += _process_streaming_chunk(chunk, writer)
            state["final_response"] = full_response
        else:
            # Non-streaming fallback: ainvoke and return complete response.
            # extract_text handles the case where include_thoughts=True
            # causes response.content to be a list of blocks instead of a string.
            response = await llm.ainvoke(prompt)
            state["final_response"] = extract_text(response.content)

        return state

    return synthesizer


# =============================================================================
# Routing Functions
# =============================================================================


def route_by_complexity(state: AnalysisState) -> Literal["react_agent", "planner", "__end__"]:
    """Route based on query complexity.

    Returns "__end__" for unclear queries — the router already set
    final_response, so no further processing is needed.
    """
    if state["query_complexity"] == "unclear":
        return END
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
    synthesizer_llm: Optional[BaseChatModel] = None,
    sec_header: str = "",
    user_id: str | None = None,
) -> StateGraph:
    """
    Create the unified LangGraph workflow with planning capabilities.

    Flow:
    1. Router classifies query complexity
    2. Simple → ReAct agent → END
    3. Complex → Planner → Step Executor (loop) → Synthesizer → END

    Args:
        llm: Main LLM for router, planner, react agent (must NOT have thinking
             enabled — Gemini's thought signatures conflict with tool calls)
        ticker: Company ticker symbol
        tavily_api_key: Optional Tavily key for research tools
        synthesizer_llm: Optional separate LLM for the synthesizer node.
             This can have thinking enabled since it doesn't call tools.
             Falls back to `llm` if not provided.
        sec_header: SEC EDGAR identity header for this session
    """
    # Create tools and planner
    tools = _create_tools(ticker, llm, tavily_api_key, sec_header, user_id=user_id)
    tools_dict = _build_tools_dict(tools)
    has_research = tavily_api_key is not None

    planner = create_planner(llm, ticker, has_research)

    # Create ReAct agent for simple queries
    system_prompt = SEC_AGENT_SYSTEM_PROMPT.format(ticker=ticker)
    react_agent = create_react_agent(llm, tools, prompt=system_prompt)

    # Build the graph
    workflow = StateGraph(AnalysisState)

    # Add nodes — synthesizer gets its own LLM (optionally with thinking)
    workflow.add_node("router", create_router_node(planner))
    workflow.add_node("react_agent", create_react_node(react_agent))
    workflow.add_node("planner", create_planner_node(planner))
    workflow.add_node("step_executor", create_step_executor_node(tools_dict, ticker))
    workflow.add_node("synthesizer", create_synthesizer_node(synthesizer_llm or llm, ticker))

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

    Three execution modes:
    - invoke(): blocks until done, returns final result (for tests, simple callers)
    - stream(): async generator using astream() — preferred for async handlers (FastAPI)
    - stream_sync(): sync generator using stream() — for testing with mocked workflows
    """

    # Human-readable descriptions for each graph node
    _NODE_MESSAGES = {
        "router": "Classifying query complexity...",
        "planner": "Planning analysis steps...",
        "step_executor": "Executing analysis step...",
        "react_agent": "Processing query...",
        "synthesizer": "Synthesizing final response...",
    }

    def __init__(
        self,
        workflow: StateGraph,
        ticker: str,
    ):
        self.workflow = workflow
        self.ticker = ticker

    def _build_initial_state(self, messages: List[BaseMessage]) -> AnalysisState:
        """Build the initial state dict. Shared by invoke() and stream_sync()."""
        return {
            "messages": messages,
            "ticker": self.ticker,
            "query_complexity": "",
            "classification": None,
            "plan": None,
            "current_step_index": 0,
            "step_results": {},
            "final_response": "",
        }

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query through the planning workflow (blocking).

        Args:
            inputs: Dict with "messages" key containing conversation history

        Returns:
            Dict with "messages" key containing updated conversation
        """
        messages = inputs.get("messages", [])
        initial_state = self._build_initial_state(messages)

        final_state = self.workflow.invoke(initial_state)

        response = final_state.get("final_response", "")
        if response:
            return {"messages": messages + [AIMessage(content=response)]}

        return {"messages": messages}

    def stream_sync(self, inputs: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Yield streaming events as the graph executes.

        Uses workflow.stream() with two stream modes:
        - "updates": emitted after each node completes → we yield node/tool events
        - "custom": emitted by get_stream_writer() inside nodes → thinking/token events

        Events yielded:
        - {"type": "node",     "node": str, "message": str}
        - {"type": "tool",     "node": str, "tool": str, "action": str, "step": int, "total": int}
        - {"type": "thinking", "message": str}
        - {"type": "token",    "message": str}
        - {"type": "response", "message": str}
        """
        messages = inputs.get("messages", [])
        initial_state = self._build_initial_state(messages)

        # Track the plan across node iterations so we can extract
        # tool info when step_executor completes
        current_plan: Optional[QueryPlan] = None

        for mode, chunk in self.workflow.stream(
            initial_state, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                # Events from get_stream_writer() (thinking, tokens)
                yield chunk

            elif mode == "updates":
                # Node completion events: chunk is {node_name: state_delta}
                for node_name, state_update in chunk.items():
                    yield {
                        "type": "node",
                        "node": node_name,
                        "message": self._NODE_MESSAGES.get(
                            node_name, f"Running {node_name}..."
                        ),
                    }

                    # Track the plan so we can look up tool names later
                    plan_update = state_update.get("plan")
                    if plan_update is not None:
                        current_plan = plan_update

                    # After step_executor, emit a tool event for every executed step.
                    # All steps run in a single node pass now (parallel by layer),
                    # so we report them all at once when the node completes.
                    if node_name == "step_executor" and current_plan:
                        for i, step in enumerate(current_plan.steps):
                            yield {
                                "type": "tool",
                                "node": node_name,
                                "tool": step.tool,
                                "action": step.action,
                                "step": i + 1,
                                "total": len(current_plan.steps),
                            }

                    # Emit final response (from react_agent or synthesizer)
                    final = state_update.get("final_response")
                    if final:
                        yield {"type": "response", "message": extract_text(final)}

    async def stream(
        self,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Async generator yielding streaming events via LangGraph's native astream().

        This is the preferred method for async handlers (FastAPI WebSocket).
        Uses astream(stream_mode="updates") which yields {node_name: state_delta}
        after each node completes. Custom events (thinking/tokens) are emitted
        via get_stream_writer() inside the synthesizer when available.

        ``config`` is a LangChain RunnableConfig forwarded to the graph. Pass
        ``metadata={"session_id": ...}`` to group runs into a LangSmith thread.
        """
        messages = inputs.get("messages", [])
        initial_state = self._build_initial_state(messages)
        current_plan: Optional[QueryPlan] = None

        async for chunk in self.workflow.astream(
            initial_state, stream_mode="updates", config=config
        ):
            # chunk is {node_name: state_delta}
            for node_name, state_update in chunk.items():
                yield {
                    "type": "node",
                    "node": node_name,
                    "message": self._NODE_MESSAGES.get(
                        node_name, f"Running {node_name}..."
                    ),
                }

                plan_update = state_update.get("plan")
                if plan_update is not None:
                    current_plan = plan_update

                if node_name == "step_executor" and current_plan:
                    for i, step in enumerate(current_plan.steps):
                        yield {
                            "type": "tool",
                            "node": node_name,
                            "tool": step.tool,
                            "action": step.action,
                            "step": i + 1,
                            "total": len(current_plan.steps),
                        }

                final = state_update.get("final_response")
                if final:
                    # Safety net: ensure response is always a string
                    yield {"type": "response", "message": extract_text(final)}


def create_planning_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
    synthesizer_llm: Optional[BaseChatModel] = None,
    sec_header: str = "",
    user_id: str | None = None,
) -> PlanningAgent:
    """
    Create a planning-enabled agent for financial analysis.

    Args:
        ticker: Company ticker symbol
        llm: Main LLM for tool-calling nodes (no thinking enabled)
        tavily_api_key: Optional Tavily API key for web research tools
        synthesizer_llm: Optional thinking-enabled LLM for synthesis only.
             Gemini's thought signatures conflict with tool calls, so thinking
             must be isolated to the synthesizer which doesn't call tools.
        sec_header: SEC EDGAR identity header for this session
        user_id: Anonymous user ID for scoping briefing queries

    Returns:
        PlanningAgent instance with invoke() and stream_sync() methods
    """
    workflow = create_planning_workflow(llm, ticker, tavily_api_key, synthesizer_llm, sec_header, user_id=user_id)
    return PlanningAgent(workflow, ticker)


# =============================================================================
# Backward Compatibility
# =============================================================================


def create_sec_qa_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
    synthesizer_llm: Optional[BaseChatModel] = None,
    sec_header: str = "",
    user_id: str | None = None,
) -> PlanningAgent:
    """
    Create a Q&A agent with planning capabilities.

    This maintains backward compatibility with existing code while
    providing the new planning functionality.
    """
    return create_planning_agent(ticker, llm, tavily_api_key, synthesizer_llm, sec_header, user_id=user_id)
