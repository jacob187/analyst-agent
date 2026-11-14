"""Skills agent: prompt assembly, skill loading, tool filtering, the WS event
contract, model-call deadlines and callback propagation.

Only the LLM boundary is faked: `ScriptedToolModel` is a chat model that
streams scripted turns (text, thinking blocks, tool calls) and records what it
was sent. Domain tools are fakes named like the real ones (the real ones hit
yfinance / SEC EDGAR / Tavily); everything else is the real agent.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import Tool
from pydantic import Field

from agents.graph import analyst_graph
from agents.graph.analyst_graph import LLMTimeoutError
from agents.graph.skills_agent import (
    LOAD_SKILL_TOOL,
    SkillsAgent,
    create_skills_graph,
)
from agents.skills import (
    SKILLS,
    SKILLS_DIR,
    known_tool_names,
    load_base_prompt,
    load_skills,
    render_base_prompt,
    render_catalogue,
)
from api import llm_concurrency
from api.llm_concurrency import LLMConcurrencyExceeded

TICKER = "AAPL"
CONTRACT_TYPES = {"node", "tool", "thinking", "token", "response"}


# ---------------------------------------------------------------------------
# LLM-boundary fake
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One scripted model response and how it streams."""

    message: AIMessage
    first_delay: float = 0.0  # before the first chunk
    gap: float = 0.0  # between chunks
    text_with_tool_call: bool = False  # put the text on the tool-call chunk


def call(name: str, args: dict | None = None, call_id: str | None = None) -> dict:
    # Single-input `Tool`s expose one required `__arg1` string to the model.
    return {"name": name, "args": args or {"__arg1": ""}, "id": call_id or f"call_{name}"}


def load(skill: str, call_id: str | None = None) -> dict:
    return call(LOAD_SKILL_TOOL, {"skill_name": skill}, call_id or f"load_{skill}")


def tool_turn(*calls: dict, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


class ScriptedToolModel(BaseChatModel):
    """Chat model that plays back scripted turns and records each request."""

    turns: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)  # [{"messages": [...], "tools": [...]}]

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(self, tools, **kwargs):
        names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        return self.bind(tool_names=names)

    def _next(self, messages, kwargs) -> Turn:
        self.requests.append({"messages": list(messages), "tools": kwargs.get("tool_names", [])})
        turn = self.turns.pop(0)
        return turn if isinstance(turn, Turn) else Turn(turn)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        turn = self._next(messages, kwargs)
        return ChatResult(generations=[ChatGeneration(message=turn.message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        turn = self._next(messages, kwargs)
        await asyncio.sleep(turn.first_delay)
        return ChatResult(generations=[ChatGeneration(message=turn.message)])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        turn = self._next(messages, kwargs)
        msg = turn.message
        content = msg.content
        if isinstance(content, str):
            pieces: list[Any] = re.findall(r"\S+\s*", content)
        else:
            pieces = [[block] for block in content]
        tool_chunks = [
            {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
            for i, tc in enumerate(msg.tool_calls)
        ]
        chunks = [AIMessageChunk(content=p) for p in pieces]
        if tool_chunks:
            if turn.text_with_tool_call and chunks:
                chunks = [AIMessageChunk(content=content, tool_call_chunks=tool_chunks)]
            else:
                chunks.append(AIMessageChunk(content="", tool_call_chunks=tool_chunks))
        if not chunks:
            chunks = [AIMessageChunk(content="")]
        await asyncio.sleep(turn.first_delay)
        for i, chunk in enumerate(chunks):
            if i:
                await asyncio.sleep(turn.gap)
            gen = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.content if isinstance(chunk.content, str) else "", chunk=gen
                )
            yield gen


def fake_tools(names=None) -> list[Tool]:
    """Stand-ins for the real tools (which hit yfinance / EDGAR / Tavily)."""
    return [
        Tool.from_function(name=n, description=f"fake {n}", func=lambda q="", n=n: f"{n} result")
        for n in sorted(names if names is not None else known_tool_names())
    ]


def build(turns, tools=None) -> tuple[ScriptedToolModel, SkillsAgent]:
    model = ScriptedToolModel(turns=list(turns))
    graph = create_skills_graph(model, TICKER, tools if tools is not None else fake_tools())
    return model, SkillsAgent(graph, TICKER)


async def collect(agent, query: str = "q", config=None) -> list[dict]:
    return [ev async for ev in agent.stream({"messages": [HumanMessage(content=query)]}, config)]


def tool_messages(request: dict) -> list[ToolMessage]:
    return [m for m in request["messages"] if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------------------
# AC3 / AC8: system prompt and load_skill
# ---------------------------------------------------------------------------


class TestPromptAssembly:
    async def test_system_prompt_lists_every_skill_name_and_description(self):
        model, agent = build([AIMessage(content="ok")])
        await collect(agent)

        system = model.requests[0]["messages"][0]
        assert isinstance(system, SystemMessage)
        for skill in SKILLS.values():
            assert f"**{skill.name}**: {skill.description}" in system.content

    async def test_system_prompt_is_the_base_file_plus_the_catalogue(self):
        model, agent = build([AIMessage(content="ok")])
        await collect(agent)

        expected = render_base_prompt(TICKER) + "\n\n" + render_catalogue(SKILLS)
        assert model.requests[0]["messages"][0].content == expected
        assert "{ticker}" not in expected and TICKER in expected

    @pytest.mark.parametrize("skill_name", sorted(SKILLS))
    async def test_load_skill_returns_the_full_instructions(self, skill_name):
        model, agent = build([tool_turn(load(skill_name)), AIMessage(content="done")])
        await collect(agent)

        (result,) = tool_messages(model.requests[1])
        assert result.content == render_base_prompt(TICKER, SKILLS[skill_name].instructions)

    async def test_prompt_assembly_reads_only_skill_files(self, monkeypatch):
        import agents.prompts.loader as yaml_loader

        def _forbidden(*a, **k):
            raise AssertionError("skills agent must not load YAML prompts")

        monkeypatch.setattr(yaml_loader, "load_prompts", _forbidden)
        read: list[Path] = []
        real_read_text = Path.read_text

        def _recording_read_text(self, *a, **k):
            read.append(Path(self).resolve())
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _recording_read_text)

        skills = load_skills()
        base = load_base_prompt()
        model = ScriptedToolModel(turns=[tool_turn(load("technicals")), AIMessage(content="done")])
        agent = SkillsAgent(create_skills_graph(model, TICKER, fake_tools(), skills), TICKER)
        await collect(agent)

        assert read, "skill files should have been read"
        skills_root = SKILLS_DIR.resolve()
        assert all(p.is_relative_to(skills_root) and p.suffix == ".md" for p in read), read
        assert model.requests[0]["messages"][0].content.startswith(base.replace("{ticker}", TICKER))

    @pytest.mark.parametrize(
        "path",
        [
            Path(analyst_graph.__file__).parent / "skills_agent.py",
            SKILLS_DIR / "__init__.py",
        ],
    )
    def test_skills_modules_never_import_yaml_prompts(self, path):
        tree = ast.parse(path.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not any(m == "agents.prompts" or m.startswith("agents.prompts.") for m in imported)


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------


class TestToolFiltering:
    async def test_only_load_skill_is_visible_before_any_skill_is_loaded(self):
        model, agent = build([AIMessage(content="ok")])
        await collect(agent)
        assert model.requests[0]["tools"] == [LOAD_SKILL_TOOL]

    @pytest.mark.parametrize("skill_name", sorted(SKILLS))
    async def test_loading_a_skill_unlocks_exactly_its_tools(self, skill_name):
        model, agent = build([tool_turn(load(skill_name)), AIMessage(content="done")])
        await collect(agent)

        assert set(model.requests[1]["tools"]) == {LOAD_SKILL_TOOL, *SKILLS[skill_name].tools}

    async def test_parallel_loads_unlock_both_skills(self):
        model, agent = build([
            tool_turn(load("technicals"), load("material-events")),
            AIMessage(content="done"),
        ])
        events = await collect(agent)

        assert set(model.requests[1]["tools"]) == {
            LOAD_SKILL_TOOL, *SKILLS["technicals"].tools, *SKILLS["material-events"].tools,
        }
        assert events[-1] == {"type": "response", "message": "done"}

    async def test_skills_accumulate_across_turns(self):
        model, agent = build([
            tool_turn(load("technicals")),
            tool_turn(load("market-context")),
            AIMessage(content="done"),
        ])
        await collect(agent)
        assert set(model.requests[2]["tools"]) == {
            LOAD_SKILL_TOOL, *SKILLS["technicals"].tools, *SKILLS["market-context"].tools,
        }

    async def test_unknown_skill_is_refused_and_unlocks_nothing(self):
        model, agent = build([tool_turn(load("astrology")), AIMessage(content="done")])
        await collect(agent)

        (result,) = tool_messages(model.requests[1])
        assert "Unknown skill 'astrology'" in result.content
        for name in SKILLS:
            assert name in result.content
        assert model.requests[1]["tools"] == [LOAD_SKILL_TOOL]

    async def test_skill_with_no_available_tools_is_hidden(self):
        no_research = known_tool_names() - set(SKILLS["news-research"].tools)
        model, agent = build(
            [tool_turn(load("news-research")), AIMessage(content="done")],
            tools=fake_tools(no_research),
        )
        await collect(agent)

        system = model.requests[0]["messages"][0].content
        assert "**news-research**" not in system
        assert "**technicals**" in system
        (result,) = tool_messages(model.requests[1])
        assert "Unknown skill 'news-research'" in result.content

    async def test_partially_available_skill_reports_missing_tools(self):
        no_research = known_tool_names() - set(SKILLS["news-research"].tools)
        model, agent = build(
            [tool_turn(load("peer-comparison")), AIMessage(content="done")],
            tools=fake_tools(no_research),
        )
        await collect(agent)

        (result,) = tool_messages(model.requests[1])
        assert result.content.endswith(
            "Not available in this session: analyze_competitors, get_industry_trends, web_search."
        )
        assert set(model.requests[1]["tools"]) == {
            LOAD_SKILL_TOOL, "get_stock_info", "get_financial_metrics",
        }


# ---------------------------------------------------------------------------
# AC5: WS event contract
# ---------------------------------------------------------------------------


class TestEventContract:
    async def test_full_run_emits_the_contract(self):
        model, agent = build([
            tool_turn(load("technicals")),
            tool_turn(call("get_stock_info"), call("get_technical_analysis")),
            AIMessage(content=[
                {"type": "thinking", "thinking": "Weighing RSI against price."},
                {"type": "text", "text": "AAPL looks "},
                {"type": "text", "text": "overbought."},
            ]),
        ])
        events = await collect(agent)

        assert {e["type"] for e in events} <= CONTRACT_TYPES
        tools = [e for e in events if e["type"] == "tool"]
        assert [(e["tool"], e["step"], e["total"]) for e in tools] == [
            (LOAD_SKILL_TOOL, 1, 1),
            ("get_stock_info", 2, 3),
            ("get_technical_analysis", 3, 3),
        ]
        for e in tools:
            assert set(e) == {"type", "node", "tool", "action", "step", "total"}
        assert tools[0]["action"] == "Loading skill: technicals"
        assert {e["node"] for e in events if e["type"] == "node"} == {"model", "tools"}
        assert [e["message"] for e in events if e["type"] == "thinking"] == [
            "Weighing RSI against price."
        ]
        assert "".join(e["message"] for e in events if e["type"] == "token") == "AAPL looks overbought."
        responses = [e for e in events if e["type"] == "response"]
        assert responses == [{"type": "response", "message": "AAPL looks overbought."}]
        assert events[-1] is responses[0] or events[-1] == responses[0]
        # Tool results reached the model.
        assert {m.content for m in tool_messages(model.requests[2])} >= {
            "get_stock_info result", "get_technical_analysis result",
        }

    async def test_text_on_a_tool_call_chunk_is_not_streamed(self):
        _, agent = build([
            Turn(tool_turn(load("technicals"), text="PREAMBLE"), text_with_tool_call=True),
            AIMessage(content="final"),
        ])
        events = await collect(agent)
        assert [e["message"] for e in events if e["type"] == "token"] == ["final"]

    async def test_llm_calls_inside_tools_are_not_streamed(self):
        inner = GenericFakeChatModel(messages=iter(["INNER analysis text"]))

        def analyse(q=""):
            return inner.invoke("analyse").content

        tools = fake_tools(known_tool_names() - {"get_risk_factors_summary"}) + [
            Tool.from_function(name="get_risk_factors_summary", description="risks", func=analyse)
        ]
        _, agent = build(
            [
                tool_turn(load("filing-analysis")),
                tool_turn(call("get_risk_factors_summary")),
                AIMessage(content="Risks summarised."),
            ],
            tools=tools,
        )
        events = await collect(agent)

        tokens = "".join(e["message"] for e in events if e["type"] == "token")
        assert "INNER" not in tokens
        assert tokens == "Risks summarised."

    async def test_empty_final_answer_still_ends_with_one_response(self):
        _, agent = build([AIMessage(content="")])
        events = await collect(agent)
        assert [e for e in events if e["type"] == "response"] == [{"type": "response", "message": ""}]
        assert events[-1]["type"] == "response"

    async def test_invoke_returns_the_final_message(self):
        _, agent = build([tool_turn(load("technicals")), AIMessage(content="AAPL is at $185")])
        query = HumanMessage(content="price?")
        result = await agent.invoke({"messages": [query]})
        assert result["messages"][0] is query
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "AAPL is at $185"

    async def test_history_is_truncated_before_the_model_sees_it(self):
        model, agent = build([AIMessage(content="ok")])
        history = []
        for i in range(40):
            history += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
        history.append(HumanMessage(content="latest"))

        [ev async for ev in agent.stream({"messages": history})]

        sent = model.requests[0]["messages"][1:]  # after the system message
        assert len(sent) == analyst_graph.MESSAGES_HISTORY_LIMIT
        assert sent[-1].content == "latest"


# ---------------------------------------------------------------------------
# AC6: deadlines, LLM slot, callbacks
# ---------------------------------------------------------------------------


class TestModelCallDeadlines:
    async def test_stall_before_first_chunk_times_out(self, monkeypatch):
        monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)
        monkeypatch.setattr(analyst_graph, "LLM_STREAM_MAX_SECONDS", 5.0)
        _, agent = build([Turn(AIMessage(content="never"), first_delay=1.0)])

        with pytest.raises(LLMTimeoutError, match="no output"):
            await collect(agent)

    async def test_slow_but_steady_stream_completes(self, monkeypatch):
        # ~0.24s total, longer than the idle deadline; every gap is under it.
        monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.1)
        monkeypatch.setattr(analyst_graph, "LLM_STREAM_MAX_SECONDS", 5.0)
        text = " ".join(f"w{i}" for i in range(9))
        _, agent = build([Turn(AIMessage(content=text), gap=0.03)])

        events = await collect(agent)
        assert events[-1] == {"type": "response", "message": text}

    async def test_total_cap_times_out_a_trickling_stream(self, monkeypatch):
        monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.1)
        monkeypatch.setattr(analyst_graph, "LLM_STREAM_MAX_SECONDS", 0.15)
        text = " ".join(f"w{i}" for i in range(40))
        _, agent = build([Turn(AIMessage(content=text), gap=0.02)])

        with pytest.raises(LLMTimeoutError, match="stream cap"):
            await collect(agent)

    async def test_non_streamed_invoke_times_out(self, monkeypatch):
        monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)
        _, agent = build([Turn(AIMessage(content="never"), first_delay=1.0)])

        with pytest.raises(LLMTimeoutError):
            await agent.invoke({"messages": [HumanMessage(content="q")]})

    async def test_model_calls_wait_for_an_llm_slot(self, monkeypatch):
        monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_CONCURRENCY", 0)
        monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_WAIT_SECONDS", 0.05)
        llm_concurrency._rebuild_semaphore()
        model, agent = build([AIMessage(content="ok")])

        with pytest.raises(LLMConcurrencyExceeded):
            await collect(agent)
        assert model.requests == []

    async def test_timed_out_call_releases_its_slot(self, monkeypatch):
        monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)
        monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_CONCURRENCY", 1)
        monkeypatch.setattr(llm_concurrency, "LLM_DISPATCH_WAIT_SECONDS", 0.2)
        llm_concurrency._rebuild_semaphore()

        _, stalled = build([Turn(AIMessage(content="never"), first_delay=1.0)])
        with pytest.raises(LLMTimeoutError):
            await collect(stalled)
        _, agent = build([AIMessage(content="ok")])
        assert (await collect(agent))[-1] == {"type": "response", "message": "ok"}


class _CountingHandler(BaseCallbackHandler):
    def __init__(self):
        self.chat_model_starts = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.chat_model_starts += 1


class TestCallbacks:
    async def test_run_callbacks_reach_every_model_call(self):
        handler = _CountingHandler()
        model, agent = build([
            tool_turn(load("technicals")),
            tool_turn(call("get_stock_info")),
            AIMessage(content="done"),
        ])
        await collect(agent, config={"callbacks": [handler]})
        assert handler.chat_model_starts == len(model.requests) == 3
