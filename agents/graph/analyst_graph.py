"""
LangGraph workflow for financial analysis with query planning.

Architecture:
- Simple queries → ReAct agent directly
- Complex queries → Planner → Step Executor → Synthesizer

Streaming:
- stream_sync() yields events as the graph executes (node transitions,
  tool calls, thinking blocks, tokens, final response)
- invoke() remains unchanged for backward compatibility
"""

import json
from typing import Dict, Any, TypedDict, Annotated, Optional, List, Literal, Generator, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

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


class StepResult(TypedDict):
    """Typed record for a single executed plan step.

    `data` is the parsed JSON payload from a structured tool (e.g. the SEC
    LLM-analysis tools, which dump Pydantic models as JSON). For tools that
    return prose, `data` is empty and `raw` carries the string. `filing_ref`
    is extracted from `data.filing_metadata` so downstream nodes (the
    reconciler) can group results by the filing they describe.
    """

    tool: str
    data: Dict[str, Any]
    raw: str
    filing_ref: Optional[str]
    error: Optional[str]


def merge_step_results(
    left: Dict[int, StepResult], right: Dict[int, StepResult]
) -> Dict[int, StepResult]:
    """Reducer for `step_results` — concurrent workers each write a unique step_id.

    Keys are unique by construction (one step_id per worker), so dict union is
    commutative and associative — the requirements LangGraph imposes on
    reducers for parallel writes.
    """
    return {**left, **right}


def _extract_filing_ref(data: Dict[str, Any]) -> Optional[str]:
    """Pull a stable filing identifier out of a tool's structured payload.

    Priority: accession_number > form_type:filing_date > filing_date > None.
    Tools that don't include `filing_metadata` (stock, market, briefing) return
    None and are skipped by the reconciler.
    """
    metadata = data.get("filing_metadata") if isinstance(data, dict) else None
    if not isinstance(metadata, dict):
        return None

    accession = metadata.get("accession_number")
    if accession:
        return str(accession)

    form_type = metadata.get("form_type")
    filing_date = metadata.get("filing_date")
    if form_type and filing_date:
        return f"{form_type}:{filing_date}"
    if filing_date:
        return str(filing_date)
    return None


class Conflict(TypedDict):
    """A cross-tool disagreement on a structured field for the same filing.

    Surfaced by the reconciler when two or more workers describe the same
    `filing_ref` but report different values for a structured field
    (e.g. `event_type`). The synthesizer is instructed to address conflicts
    explicitly rather than emitting two contradictory descriptions.
    """

    filing_ref: str
    step_ids: List[int]
    field: str
    values: List[Any]


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
    step_results: Annotated[Dict[int, StepResult], merge_step_results]
    conflicts: List[Conflict]

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
    """Convert tools list to name->Tool dict for traced execution.

    Mapping to the Tool object (not ``tool.func``) lets the executor call
    ``tool.invoke()`` later, which propagates LangChain's tracing contextvars.
    Calling ``tool.func()`` directly bypasses tracing, so per-tool LLM runs
    appear as orphan ``RunnableSequence`` spans in LangSmith and the parent
    chat trace under-counts cost by 5-15x.
    """
    return {tool.name: tool for tool in tools}


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
        state["step_results"] = {}

        return state

    return planner_node


def _build_step_result(
    tool_name: str,
    raw: str,
    error: Optional[str] = None,
) -> StepResult:
    """Wrap a tool's raw output as a typed StepResult.

    Tries to parse `raw` as JSON; if it isn't structured, `data` stays empty
    and downstream nodes fall back to the raw string. `filing_ref` is pulled
    from `data.filing_metadata` so the reconciler can group results across
    tools that hit the same filing.
    """
    data: Dict[str, Any] = {}
    if not error:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return StepResult(
        tool=tool_name,
        data=data,
        raw=raw,
        filing_ref=_extract_filing_ref(data),
        error=error,
    )


def create_worker_node(tools_dict: Dict[str, Any]):
    """Create the per-step worker node fanned out via Send.

    The worker receives a Send payload `{"step": AnalysisStep}` and runs the
    tool referenced by that step. It returns `{"step_results": {step_id: ...}}`,
    which `merge_step_results` unions into the parent state.

    Tracing context (LangChain run ids) is propagated automatically by
    LangGraph's runtime across Send invocations — no manual `contextvars`
    handling needed, unlike the previous ThreadPoolExecutor implementation.
    """

    def worker(payload: Dict[str, Any]) -> Dict[str, Any]:
        step: AnalysisStep = payload["step"]

        if step.tool not in tools_dict:
            msg = f"Tool '{step.tool}' not found"
            result = _build_step_result(step.tool, f"[ERROR: {msg}]", error=msg)
        else:
            try:
                raw = str(tools_dict[step.tool].invoke(""))
                result = _build_step_result(step.tool, raw)
            except Exception as e:
                msg = str(e)
                result = _build_step_result(step.tool, f"[ERROR: {msg}]", error=msg)

        return {"step_results": {step.id: result}}

    return worker


def dispatch_steps(state: AnalysisState) -> Union[List[Send], List[str]]:
    """Conditional edge from `planner`: fan out one Send per plan step.

    `depends_on` is intentionally ignored — every step runs in parallel in a
    single LangGraph superstep. The planner emits independent steps in
    practice; ordering is a follow-up if a real query surfaces it.

    If the plan is missing or empty, route directly to the synthesizer so the
    graph terminates cleanly instead of waiting for workers that never fire.
    """
    plan = state.get("plan")
    if not plan or not plan.steps:
        return ["synthesizer"]
    return [Send("worker", {"step": step}) for step in plan.steps]


# Structured fields the reconciler compares across step_results sharing a
# filing_ref. `event_type` is the canonical case (the GOOGL bug shape from
# #30); add new fields here when other cross-tool conflicts surface.
RECONCILABLE_FIELDS = ("event_type",)


def create_reconciler_node():
    """Create the reconciler node: groups step_results by filing_ref and flags
    structured-field disagreements between tools that hit the same filing.

    Pure Python — no LLM. The synthesizer consumes the resulting `conflicts`
    list and is prompted to surface them explicitly. This moves cross-tool
    deduplication out of prose-judgement and into deterministic code.
    """

    def reconciler(state: AnalysisState) -> Dict[str, Any]:
        groups: Dict[str, List[tuple[int, StepResult]]] = {}
        for step_id, result in (state.get("step_results") or {}).items():
            ref = result.get("filing_ref") if isinstance(result, dict) else None
            if not ref:
                continue
            groups.setdefault(ref, []).append((step_id, result))

        conflicts: List[Conflict] = []
        for ref, items in groups.items():
            if len(items) < 2:
                continue
            for field in RECONCILABLE_FIELDS:
                values = []
                step_ids = []
                seen: set = set()
                for step_id, result in items:
                    val = result.get("data", {}).get(field)
                    if val is None:
                        continue
                    step_ids.append(step_id)
                    if val not in seen:
                        seen.add(val)
                        values.append(val)
                if len(values) > 1:
                    conflicts.append(
                        Conflict(
                            filing_ref=ref,
                            step_ids=sorted(step_ids),
                            field=field,
                            values=values,
                        )
                    )

        return {"conflicts": conflicts}

    return reconciler


def _format_step_result_for_prompt(result: StepResult) -> str:
    """Render a StepResult for the synthesizer.

    Prefer the structured `data` payload (full Pydantic dump from the SEC
    analysis tools) so the synthesizer sees every field. Fall back to `raw`
    for prose-returning tools (stock, market, briefing) and error markers.
    """
    if result.get("data"):
        return json.dumps(result["data"], indent=2, default=str)
    return result.get("raw", "[No result]")


def _format_conflicts_for_prompt(conflicts: List[Conflict]) -> str:
    """Render the reconciler's conflict list as a CONFLICTS DETECTED block.

    Empty when no conflicts — the synthesizer prompt absorbs the empty string
    cleanly so the existing input_variables list doesn't change.
    """
    if not conflicts:
        return ""

    lines = ["", "=== CONFLICTS DETECTED ==="]
    for c in conflicts:
        lines.append(
            f"Filing {c['filing_ref']} (steps {c['step_ids']}): "
            f"{c['field']} disagreement — {c['values']}"
        )
    return "\n".join(lines) + "\n"


def _build_synthesis_prompt(state: AnalysisState, ticker: str) -> str:
    """Build the synthesis prompt from plan state. Shared by streaming and non-streaming paths."""
    plan = state["plan"]
    step_results = state["step_results"]
    conflicts = state.get("conflicts") or []
    query = _get_latest_query(state["messages"])

    results_text = []
    if plan:
        for step in plan.steps:
            result = step_results.get(step.id)
            rendered = (
                _format_step_result_for_prompt(result) if result else "[No result]"
            )
            results_text.append(
                f"=== Step {step.id}: {step.action} (via {step.tool}) ===\n{rendered}\n"
            )

    synthesis_approach = plan.synthesis_approach if plan else "Combine findings into a comprehensive answer"

    return SYNTHESIS_SYSTEM_PROMPT.format(
        ticker=ticker,
        step_results="\n".join(results_text) + _format_conflicts_for_prompt(conflicts),
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
    3. Complex → Planner → dispatch_steps (Send fan-out) → Worker (×N) → Reconciler → Synthesizer → END

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
    workflow.add_node("worker", create_worker_node(tools_dict))
    workflow.add_node("reconciler", create_reconciler_node())
    workflow.add_node("synthesizer", create_synthesizer_node(synthesizer_llm or llm, ticker))

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_conditional_edges("router", route_by_complexity)
    workflow.add_edge("react_agent", END)
    # Planner fans out one Send per plan step → worker (parallel within one
    # superstep). If the plan is empty, dispatch_steps routes straight to
    # synthesizer instead, bypassing the reconciler.
    workflow.add_conditional_edges(
        "planner", dispatch_steps, ["worker", "synthesizer"]
    )
    # All workers join at the reconciler, then synthesize. The reconciler is
    # pure Python (no LLM) — it groups step_results by filing_ref and emits a
    # structured `conflicts` list for the synthesizer to surface explicitly.
    workflow.add_edge("worker", "reconciler")
    workflow.add_edge("reconciler", "synthesizer")
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
        "worker": "Executing analysis step...",
        "reconciler": "Reconciling step results...",
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
            "step_results": {},
            "conflicts": [],
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

    def _worker_tool_events(
        self,
        state_update: Dict[str, Any],
        plan: Optional[QueryPlan],
    ) -> Generator[Dict[str, Any], None, None]:
        """Convert a worker node's state delta into one tool event per step.

        Send-based fan-out emits one updates-mode chunk per worker. Each
        worker's `step_results` delta carries exactly one (step_id → result)
        entry — but iterating defensively handles the case where multiple
        workers' updates get coalesced into a single emission.
        """
        if not plan:
            return
        delta_results: Dict[int, Any] = state_update.get("step_results") or {}
        steps_by_id = {s.id: s for s in plan.steps}
        total = len(plan.steps)
        for step_id in delta_results:
            step = steps_by_id.get(step_id)
            if step is None:
                continue
            yield {
                "type": "tool",
                "node": "worker",
                "tool": step.tool,
                "action": step.action,
                "step": step.id,
                "total": total,
            }

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

        # Track the plan so worker-update deltas can be mapped back to step metadata
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

                    # Per-step UX: each Send-fanned worker emission becomes
                    # one tool event tagged with its step's metadata.
                    if node_name == "worker":
                        yield from self._worker_tool_events(state_update, current_plan)

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

                if node_name == "worker":
                    for event in self._worker_tool_events(state_update, current_plan):
                        yield event

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
