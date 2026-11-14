"""AGENT_ARCHITECTURE flag and the skills agent behind the real WS route.

The route, DB (temp SQLite), budget charger, free-trial meter and tracing are
all real; only the chat model is the scripted fake from test_skills_agent.
Tools the scripts call need no network: `load_skill`, `get_briefing_history`
for an anonymous user (returns "no user context"), and a trial-metered
research tool whose meter refuses (returns the exhausted notice).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from agents.graph import agent_architecture, build_agent
from agents.graph import analyst_graph
from agents.graph.analyst_graph import PlanningAgent
from agents.graph.skills_agent import LOAD_SKILL_TOOL, SkillsAgent
from agents.tools.research_tools import TRIAL_SEARCH_EXHAUSTED
from api.routes import chat
from api.tracing import capture_spans
from evals.benchmark.architectures import build_skills, get_builder
from tests.test_chat_ws import _drain_until, _usage, _wait_for_usage, client, temp_db  # noqa: F401
from tests.test_skills_agent import ScriptedToolModel, Turn, call, load, tool_turn

KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_KIND = OpenInferenceSpanKindValues.LLM.value
TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value

_LENDABLE = "gemini-3.6-flash"
_SIGNED_IN = "user_skillsbudget"


def _fake_llm():
    # The current graph's planner needs with_structured_output; constructing a
    # real provider model makes no network call.
    return ChatOpenAI(model="gpt-4o-mini", api_key="not-a-real-key")


# ---------------------------------------------------------------------------
# AC1: the flag
# ---------------------------------------------------------------------------


class TestArchitectureFlag:
    def test_default_is_current(self, monkeypatch):
        monkeypatch.delenv("AGENT_ARCHITECTURE", raising=False)
        assert agent_architecture() == "current"

    @pytest.mark.parametrize("value, expected", [("skills", "skills"), (" Skills ", "skills"), ("current", "current")])
    def test_env_selects_architecture(self, monkeypatch, value, expected):
        monkeypatch.setenv("AGENT_ARCHITECTURE", value)
        assert agent_architecture() == expected

    def test_unknown_value_is_refused(self, monkeypatch):
        monkeypatch.setenv("AGENT_ARCHITECTURE", "swarm")
        with pytest.raises(ValueError, match="swarm"):
            agent_architecture()

    def test_build_agent_returns_each_architecture(self):
        assert isinstance(build_agent("current", "AAPL", _fake_llm()), PlanningAgent)
        assert isinstance(build_agent("skills", "AAPL", _fake_llm()), SkillsAgent)
        with pytest.raises(ValueError):
            build_agent("swarm", "AAPL", _fake_llm())

    def test_benchmark_registers_skills(self):
        assert get_builder("skills") is build_skills


def _wire(monkeypatch, turns, architecture="skills") -> ScriptedToolModel:
    model = ScriptedToolModel(turns=list(turns))
    monkeypatch.setattr(chat, "create_llm_pair", lambda *a, **k: (model, model))
    if architecture is None:
        monkeypatch.delenv("AGENT_ARCHITECTURE", raising=False)
    else:
        monkeypatch.setenv("AGENT_ARCHITECTURE", architecture)
    return model


def _ask(client, auth: dict, query: str = "how are briefings looking?") -> list[dict]:
    """Send one query; return every frame up to and including the response or error."""
    frames = []
    with client.websocket_connect("/ws/chat/AAPL") as ws:
        ws.send_text(json.dumps(auth))
        _drain_until(ws, "auth_success")
        _drain_until(ws, "system")  # "agent initialized" notice, sent by the route
        ws.send_text(json.dumps({"type": "query", "message": query}))
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame["type"] in ("response", "error"):
                return frames


_BYOK = {"type": "auth", "google_api_key": "byok-key"}

_BRIEFING_SCRIPT = [
    tool_turn(load("market-context")),
    tool_turn(call("get_briefing_history", {"__arg1": "7"})),
    AIMessage(content="No briefings yet."),
]


class TestRouteUsesTheFlag:
    def test_default_flag_builds_the_current_agent(self, client, temp_db, monkeypatch):
        built = []

        class _Agent:
            def stream(self, *a, **k):
                async def _gen():
                    yield {"type": "response", "message": "current answer"}
                return _gen()

        _wire(monkeypatch, [], architecture=None)
        monkeypatch.setattr(
            analyst_graph, "create_sec_qa_agent", lambda *a, **k: built.append(a) or _Agent()
        )
        frames = _ask(client, _BYOK)
        assert frames[-1] == {"type": "response", "message": "current answer"}
        assert len(built) == 1

    def test_skills_flag_answers_with_the_skills_agent(self, client, temp_db, monkeypatch):
        model = _wire(monkeypatch, _BRIEFING_SCRIPT)
        frames = _ask(client, _BYOK)

        assert frames[-1] == {"type": "response", "message": "No briefings yet."}
        assert {f["type"] for f in frames} <= {"node", "tool", "thinking", "token", "response"}
        assert [f["tool"] for f in frames if f["type"] == "tool"] == [
            LOAD_SKILL_TOOL, "get_briefing_history",
        ]
        (briefing,) = [m for m in model.requests[2]["messages"] if isinstance(m, ToolMessage)
                       and m.name == "get_briefing_history"]
        assert "no user context" in briefing.content


# ---------------------------------------------------------------------------
# AC6: budget, free trial, tracing, timeouts through the route
# ---------------------------------------------------------------------------


class TestBudget:
    def test_operator_paid_query_charges_each_model_call(self, client, temp_db, monkeypatch):
        model = _wire(monkeypatch, [tool_turn(load("technicals")), AIMessage(content="Done.")])
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")

        frames = _ask(client, {"type": "auth", "user_id": _SIGNED_IN, "model_id": _LENDABLE})

        assert frames[-1] == {"type": "response", "message": "Done."}
        assert len(model.requests) == 2
        assert _wait_for_usage(_SIGNED_IN, 2) == 2

    def test_byok_query_charges_nothing(self, client, temp_db, monkeypatch):
        _wire(monkeypatch, [tool_turn(load("technicals")), AIMessage(content="Done.")])
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")

        _ask(client, {"type": "auth", "user_id": _SIGNED_IN, "model_id": _LENDABLE,
                      "google_api_key": "their-key"})
        assert _usage(_SIGNED_IN) == 0


class TestFreeTrial:
    @pytest.fixture(autouse=True)
    def _clerk_on(self, monkeypatch):
        # Clerk configured, no token: an anonymous caller can only reach the
        # operator's key through the free-trial carve-out.
        monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")

    def test_trial_query_is_counted_and_research_goes_through_the_meter(
        self, client, temp_db, monkeypatch
    ):
        import api.db as db_module

        model = _wire(monkeypatch, [
            tool_turn(load("news-research")),
            tool_turn(call("web_search", {"__arg1": "skills-trial-unique-query"})),
            AIMessage(content="Search is unavailable."),
        ])
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")
        monkeypatch.setenv("TAVILY_API_KEY", "operator-tavily")

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps({"type": "auth"}))
            ok = _drain_until(ws, "auth_success")
            assert ok["free_trial"] is True
            assert "Web research enabled" in ok["message"]
            # Spend today's search allowance before the query: the meter must refuse.
            monkeypatch.setattr(chat, "ANON_FREE_SEARCH_DAILY_CAP", 0)
            ws.send_text(json.dumps({"type": "query", "message": "latest news?"}))
            assert _drain_until(ws, "response")["message"] == "Search is unavailable."

        (search,) = [m for m in model.requests[2]["messages"]
                     if isinstance(m, ToolMessage) and m.name == "web_search"]
        assert search.content == TRIAL_SEARCH_EXHAUSTED
        assert asyncio.run(db_module.get_llm_usage(chat.FREE_TRIAL_SEARCH_USAGE_KEY)) == 1
        assert asyncio.run(db_module.get_llm_usage(chat.FREE_TRIAL_GLOBAL_USAGE_KEY)) == 1

    def test_trial_quota_still_refuses_with_the_skills_agent(self, client, temp_db, monkeypatch):
        _wire(monkeypatch, [AIMessage(content="One.")])
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")
        monkeypatch.setattr(chat, "ANON_FREE_QUERIES", 1)

        with client.websocket_connect("/ws/chat/AAPL") as ws:
            ws.send_text(json.dumps({"type": "auth"}))
            _drain_until(ws, "auth_success")
            ws.send_text(json.dumps({"type": "query", "message": "one"}))
            assert _drain_until(ws, "response")["message"] == "One."
            ws.send_text(json.dumps({"type": "query", "message": "two"}))
            assert "Free trial limit reached" in _drain_until(ws, "error")["message"]


def _by_id(spans):
    return {s.context.span_id: s for s in spans}


def _root_of(span, by_id):
    while span.parent is not None:
        span = by_id[span.parent.span_id]
    return span


class TestTracing:
    def test_model_and_tool_spans_descend_from_chat_query(self, client, temp_db, monkeypatch):
        _wire(monkeypatch, _BRIEFING_SCRIPT)
        with capture_spans() as captured:
            frames = _ask(client, _BYOK)
        assert frames[-1]["message"] == "No briefings yet."

        spans = captured.spans
        by_id = _by_id(spans)
        roots = [s for s in spans if s.parent is None]
        assert [r.name for r in roots] == ["chat.query"]
        for s in spans:
            assert _root_of(s, by_id) is roots[0]
        assert sum(1 for s in spans if s.attributes.get(KIND) == LLM_KIND) == 3
        tool_spans = {s.name for s in spans if s.attributes.get(KIND) == TOOL_KIND}
        assert {LOAD_SKILL_TOOL, "get_briefing_history"} <= tool_spans


class TestTimeoutOverWs:
    def test_stalled_model_reports_the_timeout_message(self, client, temp_db, monkeypatch):
        monkeypatch.setattr(analyst_graph, "LLM_CALL_TIMEOUT", 0.05)
        _wire(monkeypatch, [Turn(AIMessage(content="never"), first_delay=1.0)])

        frames = _ask(client, _BYOK)
        assert frames[-1]["type"] == "error"
        assert frames[-1]["message"].startswith("The model took too long to respond.")
