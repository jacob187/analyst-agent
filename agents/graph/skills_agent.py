"""Skills-based single agent: one tool-calling loop guided by on-demand skills.

The alternative to the router/planner/synthesizer graph in ``analyst_graph``.
One model runs a ReAct loop (``langchain.agents.create_agent``); a
``SkillsMiddleware`` shows it a catalogue of skills (``agents/skills/``) and
every tool; ``load_skill`` returns a skill's instructions on demand. Set
``SKILLS_FILTER_TOOLS=true`` to expose only the tools of loaded skills instead.

Model choice: the whole loop runs on the thinking-enabled LLM from
``create_llm_pair`` (medium thinking, thoughts included). The same model now
both picks tools and writes the final answer the current graph's synthesizer
writes, so it gets the synthesizer's reasoning budget; thoughts stream to the
client as ``thinking`` events as they do today. Gemini 3 thought signatures
are round-tripped by ``langchain-google-genai`` 4.x across tool calls.

Every model call holds a process-wide LLM slot and runs under the same
deadlines as the current graph: ``LLM_CALL_TIMEOUT`` as an idle deadline
(reset by each streamed token) and ``LLM_STREAM_MAX_SECONDS`` as a total cap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextvars import ContextVar
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Mapping, NotRequired, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tracers.context import register_configure_hook
from langgraph.types import Command

from agents.graph import analyst_graph
from agents.graph.analyst_graph import LLMTimeoutError, _create_tools, _truncate_messages
from agents.llm_utils import extract_text, parse_llm_response
from agents.skills import SKILLS, Skill, render_base_prompt, render_catalogue
from api.llm_concurrency import llm_slot

logger = logging.getLogger(__name__)

LOAD_SKILL_TOOL = "load_skill"
FILTER_TOOLS_ENV = "SKILLS_FILTER_TOOLS"


def filter_tools_enabled() -> bool:
    """``SKILLS_FILTER_TOOLS=true`` restores the old behaviour (tools unlock per loaded skill)."""
    return os.getenv(FILTER_TOOLS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# Model-call deadlines
# =============================================================================


class _ModelActivity(BaseCallbackHandler):
    """Records when the in-flight model call last produced a token."""

    run_inline = True  # update the timestamp as the chunk arrives, not via an executor

    def __init__(self) -> None:
        self.last = time.monotonic()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last = time.monotonic()


# Set only while one model call runs; LangChain adds the handler to any
# callback manager configured in that context, i.e. the chat model's run.
_activity_var: ContextVar[Optional[_ModelActivity]] = ContextVar(
    "skills_agent_model_activity", default=None
)
register_configure_hook(_activity_var, inheritable=False)


async def _call_model_with_deadlines(call: Callable[[], Awaitable[Any]], *, label: str) -> Any:
    """Run one model call in an LLM slot under an idle deadline and a total cap.

    Matches ``_run_with_timeout`` / ``_stream_with_timeout``: a non-streamed
    call must finish within ``LLM_CALL_TIMEOUT``; a streamed call must produce
    a token at least every ``LLM_CALL_TIMEOUT`` seconds and finish within
    ``LLM_STREAM_MAX_SECONDS``.
    """
    idle_timeout = analyst_graph.LLM_CALL_TIMEOUT
    max_total = analyst_graph.LLM_STREAM_MAX_SECONDS
    async with llm_slot():
        activity = _ModelActivity()
        token = _activity_var.set(activity)
        try:
            task = asyncio.ensure_future(call())
        finally:
            _activity_var.reset(token)
        deadline = time.monotonic() + max_total
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise LLMTimeoutError(
                        f"LLM call '{label}' exceeded the {max_total}s stream cap"
                    )
                idle_left = activity.last + idle_timeout - now
                if idle_left <= 0:
                    logger.warning("LLM call stalled: label=%s idle_timeout=%ss", label, idle_timeout)
                    raise LLMTimeoutError(
                        f"LLM call '{label}' produced no output for {idle_timeout:.0f}s"
                    )
                done, _ = await asyncio.wait({task}, timeout=min(idle_left, deadline - now))
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
            if not task.cancelled():
                task.exception()  # mark retrieved; the result path already returned it


# =============================================================================
# Skills middleware
# =============================================================================


def _merge_skill_names(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Reducer: union of loaded skill names, first-load order, no duplicates.

    Parallel ``load_skill`` calls in one turn each write this key in the same
    step, so it needs a reducer rather than last-value semantics.
    """
    return list(dict.fromkeys([*(left or []), *(right or [])]))


class SkillsState(AgentState):
    skills_loaded: NotRequired[Annotated[List[str], _merge_skill_names]]


def _make_load_skill_tool(
    skills: Mapping[str, Skill], available_tools: set[str], ticker: str
) -> BaseTool:
    valid_names = ", ".join(skills)

    @tool(LOAD_SKILL_TOOL)
    def load_skill(skill_name: str, runtime: ToolRuntime) -> Command | str:
        """Load a skill by name: returns its instructions for using its tools.

        Pick skill names from the skills catalogue in the system prompt. You can
        load more than one skill.
        """
        skill = skills.get(skill_name.strip())
        if skill is None:
            return f"Unknown skill '{skill_name}'. Available skills: {valid_names}."
        body = render_base_prompt(ticker, skill.instructions)
        missing = [t for t in skill.tools if t not in available_tools]
        if missing:
            body += f"\n\nNot available in this session: {', '.join(missing)}."
        return Command(
            update={
                "messages": [ToolMessage(content=body, tool_call_id=runtime.tool_call_id)],
                "skills_loaded": [skill.name],
            }
        )

    return load_skill


class SkillsMiddleware(AgentMiddleware[SkillsState]):
    """Adds the skill catalogue to the system message; optionally filters tools.

    By default the model always sees every tool plus ``load_skill``; skills only
    supply instructions on demand. With ``SKILLS_FILTER_TOOLS=true`` it sees just
    ``load_skill`` until ``load_skill("<name>")`` unlocks that skill's tools. Skills
    none of whose tools exist in this session (e.g. research without a Tavily
    key) are left out of the catalogue.
    """

    state_schema = SkillsState

    def __init__(
        self,
        skills: Mapping[str, Skill],
        available_tools: set[str],
        ticker: str,
        filter_tools: Optional[bool] = None,
    ):
        super().__init__()
        self.filter_tools = filter_tools_enabled() if filter_tools is None else filter_tools
        self.skills = {
            name: skill
            for name, skill in skills.items()
            if any(t in available_tools for t in skill.tools)
        }
        self.catalogue = render_catalogue(self.skills, filter_tools=self.filter_tools)
        self.tools = [_make_load_skill_tool(self.skills, available_tools, ticker)]

    def _prepare(self, request: Any) -> Any:
        base = request.system_message.text if request.system_message else ""
        system = f"{base}\n\n{self.catalogue}" if base else self.catalogue
        if not self.filter_tools:
            return request.override(system_message=SystemMessage(content=system))
        loaded = request.state.get("skills_loaded") or []
        allowed = {LOAD_SKILL_TOOL}
        for name in loaded:
            if name in self.skills:
                allowed.update(self.skills[name].tools)
        tools = [t for t in request.tools if not isinstance(t, dict) and t.name in allowed]
        return request.override(tools=tools, system_message=SystemMessage(content=system))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        prepared = self._prepare(request)
        return await _call_model_with_deadlines(lambda: handler(prepared), label="skills_agent")


# =============================================================================
# Agent wrapper
# =============================================================================


def create_skills_graph(
    model: BaseChatModel,
    ticker: str,
    tools: List[BaseTool],
    skills: Optional[Mapping[str, Skill]] = None,
) -> Any:
    """Compile the skills agent over ``tools`` (async-only: every model call is
    deadline-wrapped in ``awrap_model_call``)."""
    middleware = SkillsMiddleware(
        SKILLS if skills is None else skills, {t.name for t in tools}, ticker
    )
    return create_agent(
        model,
        tools=tools,
        system_prompt=render_base_prompt(ticker),
        middleware=[middleware],
    )


def _final_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return extract_text(msg.content)
    return ""


class SkillsAgent:
    """Same interface as ``PlanningAgent``: ``invoke()`` and ``stream()``."""

    _NODE_MESSAGES = {
        "model": "Analyzing...",
        "tools": "Reviewing tool results...",
    }

    def __init__(self, graph: Any, ticker: str):
        self.graph = graph
        self.ticker = ticker

    async def invoke(
        self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        messages = inputs.get("messages", [])
        result = await self.graph.ainvoke(
            {"messages": _truncate_messages(messages)}, config=config
        )
        response = _final_text(result.get("messages", []))
        if response:
            return {"messages": messages + [AIMessage(content=response)]}
        return {"messages": messages}

    async def stream(
        self,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Yield the WS event contract (see ``PlanningAgent.stream``).

        Tokens and thinking come from the model node's message chunks (LLM
        calls made inside tools are not streamed); ``tool`` events are emitted
        as the model requests each call; one ``response`` event with the full
        final text ends the stream.
        """
        messages = inputs.get("messages", [])
        step = 0
        total = 0
        final = ""

        async for mode, chunk in self.graph.astream(
            {"messages": _truncate_messages(messages)},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                msg, metadata = chunk
                if metadata.get("langgraph_node") != "model" or not isinstance(msg, AIMessageChunk):
                    continue
                parsed = parse_llm_response(msg)
                if parsed.thinking:
                    yield {"type": "thinking", "message": parsed.thinking}
                if parsed.text and not msg.tool_call_chunks:
                    yield {"type": "token", "message": parsed.text}
                continue

            for node_name, update in chunk.items():
                yield {
                    "type": "node",
                    "node": node_name,
                    "message": self._NODE_MESSAGES.get(node_name, f"Running {node_name}..."),
                }
                if node_name != "model" or not isinstance(update, dict):
                    continue
                for msg in update.get("messages") or []:
                    if not isinstance(msg, AIMessage):
                        continue
                    if not msg.tool_calls:
                        final = extract_text(msg.content)
                        continue
                    total += len(msg.tool_calls)
                    for call in msg.tool_calls:
                        step += 1
                        yield {
                            "type": "tool",
                            "node": "tools",
                            "tool": call["name"],
                            "action": _describe_call(call),
                            "step": step,
                            "total": total,
                        }

        yield {"type": "response", "message": final}


def _describe_call(call: Dict[str, Any]) -> str:
    if call["name"] == LOAD_SKILL_TOOL:
        return f"Loading skill: {call.get('args', {}).get('skill_name', '')}"
    return f"Running {call['name']}"


def create_skills_agent(
    ticker: str,
    llm: BaseChatModel,
    tavily_api_key: Optional[str] = None,
    synthesizer_llm: Optional[BaseChatModel] = None,
    user_id: str | None = None,
    search_budget: Optional[Callable[[], Awaitable[bool]]] = None,
) -> SkillsAgent:
    """Build the skills agent; same arguments as ``create_planning_agent``.

    Tools are built exactly as the current graph builds them (SEC tools use
    ``llm`` for their in-tool analyses). The loop itself runs on
    ``synthesizer_llm`` when given — see the module docstring.
    """
    tools = _create_tools(ticker, llm, tavily_api_key, user_id=user_id, search_budget=search_budget)
    graph = create_skills_graph(synthesizer_llm or llm, ticker, tools)
    return SkillsAgent(graph, ticker)
