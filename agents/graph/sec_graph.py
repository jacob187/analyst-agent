"""
LangGraph workflow for SEC financial analysis with query planning.

Architecture:
- Simple queries → ReAct agent directly
- Complex queries → Planner → Step Executor (loop) → Synthesizer

Streaming:
- stream_sync() yields events as the graph executes (node transitions,
  tool calls, thinking blocks, tokens, final response)
- invoke() remains unchanged for backward compatibility
"""

from typing import Dict, Any, TypedDict, Annotated, Optional, List, Literal, Generator
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
            # _extract_text_content handles the case where Gemini 3 returns
            # content as a list of blocks instead of a plain string
            response = _extract_text_content(result["messages"][-1].content)
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


def _extract_text_content(content) -> str:
    """Extract plain text from an LLM response content field.

    With include_thoughts=True, Gemini returns content as a list of blocks:
    [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
    This helper extracts just the text portions, ignoring thinking blocks.
    If content is already a string, returns it as-is.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _process_streaming_chunk(chunk, writer) -> str:
    """Process a single LLM streaming chunk, emitting events via writer.

    Handles two content formats from Gemini:
    - str: regular text token (most common during streaming)
    - list[dict]: structured blocks with thinking/reasoning or text parts
      (appears when include_thoughts=True is set on the LLM)

    Returns the text content extracted from this chunk (for accumulation).
    """
    content = chunk.content
    text_content = ""

    if isinstance(content, str) and content:
        text_content = content
        writer({"type": "token", "message": content})
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type in ("thinking", "reasoning"):
                thinking_text = block.get(block_type, "")
                if thinking_text:
                    writer({"type": "thinking", "message": thinking_text})
            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    text_content += text
                    writer({"type": "token", "message": text})

    return text_content


def create_synthesizer_node(llm: BaseChatModel, ticker: str):
    """Create node that synthesizes all step results into final response.

    Supports two execution modes:
    - Streaming (via workflow.stream): uses get_stream_writer() to emit
      thinking blocks and tokens in real-time as the LLM generates them.
    - Non-streaming (via workflow.invoke): falls back to llm.invoke() for
      the complete response at once. This keeps backward compatibility.
    """

    def synthesizer(state: AnalysisState) -> AnalysisState:
        prompt = _build_synthesis_prompt(state, ticker)

        # Try to get a stream writer — only available inside workflow.stream()
        writer = None
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            pass

        if writer:
            # Streaming path: emit tokens/thinking as they arrive
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += _process_streaming_chunk(chunk, writer)
            state["final_response"] = full_response
        else:
            # Non-streaming fallback: invoke and return complete response.
            # _extract_text_content handles the case where include_thoughts=True
            # causes response.content to be a list of blocks instead of a string.
            response = llm.invoke(prompt)
            state["final_response"] = _extract_text_content(response.content)

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
    synthesizer_llm: Optional[BaseChatModel] = None,
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

                    # After step_executor, emit which tool ran and progress
                    if node_name == "step_executor" and current_plan:
                        # current_step_index was incremented by the node,
                        # so the step that just ran is index - 1
                        step_idx = state_update.get("current_step_index", 1) - 1
                        if 0 <= step_idx < len(current_plan.steps):
                            step = current_plan.steps[step_idx]
                            yield {
                                "type": "tool",
                                "node": node_name,
                                "tool": step.tool,
                                "action": step.action,
                                "step": step_idx + 1,
                                "total": len(current_plan.steps),
                            }

                    # Emit final response (from react_agent or synthesizer)
                    final = state_update.get("final_response")
                    if final:
                        yield {"type": "response", "message": _extract_text_content(final)}

    async def stream(self, inputs: Dict[str, Any]):
        """
        Async generator yielding streaming events via LangGraph's native astream().

        This is the preferred method for async handlers (FastAPI WebSocket).
        Uses astream(stream_mode="updates") which yields {node_name: state_delta}
        after each node completes. Custom events (thinking/tokens) are emitted
        via get_stream_writer() inside the synthesizer when available.
        """
        messages = inputs.get("messages", [])
        initial_state = self._build_initial_state(messages)
        current_plan: Optional[QueryPlan] = None

        async for chunk in self.workflow.astream(
            initial_state, stream_mode="updates"
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
                    step_idx = state_update.get("current_step_index", 1) - 1
                    if 0 <= step_idx < len(current_plan.steps):
                        step = current_plan.steps[step_idx]
                        yield {
                            "type": "tool",
                            "node": node_name,
                            "tool": step.tool,
                            "action": step.action,
                            "step": step_idx + 1,
                            "total": len(current_plan.steps),
                        }

                final = state_update.get("final_response")
                if final:
                    # Safety net: ensure response is always a string
                    yield {"type": "response", "message": _extract_text_content(final)}


def create_planning_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
    synthesizer_llm: Optional[BaseChatModel] = None,
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

    Returns:
        PlanningAgent instance with invoke() and stream_sync() methods
    """
    workflow = create_planning_workflow(llm, ticker, tavily_api_key, synthesizer_llm)
    return PlanningAgent(workflow, ticker)


# =============================================================================
# Backward Compatibility
# =============================================================================


def create_sec_qa_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
    synthesizer_llm: Optional[BaseChatModel] = None,
) -> PlanningAgent:
    """
    Create a Q&A agent with planning capabilities.

    This maintains backward compatibility with existing code while
    providing the new planning functionality.
    """
    return create_planning_agent(ticker, llm, tavily_api_key, synthesizer_llm)
