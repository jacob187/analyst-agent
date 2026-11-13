"""OpenTelemetry tracing: setup gating, span structure, token counts, redaction.

Real SDK throughout: spans are read back from the SDK's in-memory exporter via
`api.tracing.capture_spans()`. Only the LLM boundary is faked (a scripted chat
model, or a real `ChatOpenAI` whose network call is stubbed out).
"""

import json
import logging
import os

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agents.graph.analyst_graph import (
    AnalysisState,
    LLMTimeoutError,
    PlanningAgent,
    create_react_node,
)
from api import tracing
from api.main import app
from api.routes import chat
from api.tracing import capture_spans, get_tracer, setup_tracing, shutdown_tracing
from langgraph.prebuilt import create_react_agent

# Fixtures/helpers shared with the chat WS tests (temp SQLite DB, frame draining).
from tests.test_chat_ws import _drain_until, client, temp_db  # noqa: F401

KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_KIND = OpenInferenceSpanKindValues.LLM.value
TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
CHAIN_KIND = OpenInferenceSpanKindValues.CHAIN.value

_TRACING_ENV = [
    "OTEL_SDK_DISABLED",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TIMEOUT",
    "OTEL_SERVICE_NAME",
    "OTEL_TRACES_SAMPLER",
    "OTEL_TRACES_SAMPLER_ARG",
    "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OPENINFERENCE_HIDE_INPUTS",
    "OPENINFERENCE_HIDE_OUTPUTS",
]


@pytest.fixture(autouse=True)
def clean_tracing(monkeypatch):
    """Start every test with tracing unconfigured; undo anything it installs."""
    # setenv-then-delenv registers the original state for restoration, so a var
    # that api.tracing writes (the 8192 default) is removed again at teardown.
    for name in {*_TRACING_ENV, *(k for k in os.environ if k.startswith(("OTEL_", "OPENINFERENCE_")))}:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    yield
    shutdown_tracing(tracing._provider)
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
    assert tracing._provider is None and tracing._capture_provider is None


@pytest.fixture
def otlp_endpoint(monkeypatch):
    """An OTLP endpoint nobody listens on; a short timeout keeps shutdown quick."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")


# ── AC1: off unless configured ──────────────────────────────────────────────


class TestTracingOff:
    def test_no_endpoint_means_no_provider_and_logs_off(self, caplog):
        with caplog.at_level(logging.INFO, logger="api.tracing"):
            provider = setup_tracing()

        assert provider is None
        assert "Tracing: OFF" in caplog.text
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert tracing._provider is None

    def test_sdk_disabled_wins_over_endpoint(self, otlp_endpoint, monkeypatch, caplog):
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        with caplog.at_level(logging.INFO, logger="api.tracing"):
            provider = setup_tracing()

        assert provider is None
        assert "Tracing: OFF" in caplog.text
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry

    def test_get_tracer_is_a_working_noop_when_off(self):
        assert setup_tracing() is None
        tracer = get_tracer()
        with tracer.start_as_current_span("chat.query", attributes={"ticker": "AAPL"}) as span:
            assert not span.is_recording()

    def test_shutdown_none_is_noop(self):
        shutdown_tracing(None)


# ── AC2: on when an OTLP endpoint is configured ─────────────────────────────


class TestTracingOn:
    def test_endpoint_installs_provider_and_instrumentation(self, otlp_endpoint, monkeypatch, caplog):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "analyst-agent-prod")
        monkeypatch.setenv("OTEL_TRACES_SAMPLER", "traceidratio")
        monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "1.0")
        with caplog.at_level(logging.INFO, logger="api.tracing"):
            provider = setup_tracing()

        assert provider is not None
        assert provider.resource.attributes["service.name"] == "analyst-agent-prod"
        assert LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert "Tracing: ON service=analyst-agent-prod sampler=TraceIdRatioBased{1.0}" in caplog.text

        # The instrumentation really feeds this provider.
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        RunnableLambda(lambda x: x, name="probe").invoke("hi")
        assert any(s.name == "probe" for s in exporter.get_finished_spans())

    def test_traces_specific_endpoint_also_enables(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:9/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
        assert setup_tracing() is not None

    def test_service_name_defaults(self, otlp_endpoint, caplog):
        with caplog.at_level(logging.INFO, logger="api.tracing"):
            provider = setup_tracing()
        assert provider.resource.attributes["service.name"] == "analyst-agent"
        assert "service=analyst-agent " in caplog.text

    def test_setup_is_idempotent(self, otlp_endpoint):
        first = setup_tracing()
        assert setup_tracing() is first

    def test_get_tracer_records_spans_when_on(self, otlp_endpoint):
        provider = setup_tracing()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with get_tracer().start_as_current_span("chat.query"):
            pass
        assert [s.name for s in exporter.get_finished_spans()] == ["chat.query"]

    def test_shutdown_uninstruments_and_allows_setup_again(self, otlp_endpoint):
        provider = setup_tracing()
        shutdown_tracing(provider)
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert setup_tracing() is not provider

    def test_lifespan_sets_up_and_shuts_down(self, otlp_endpoint, monkeypatch, caplog):
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
        monkeypatch.delenv("DISABLE_AUTH", raising=False)
        with caplog.at_level(logging.INFO, logger="api.tracing"):
            with TestClient(app) as test_client:
                assert test_client.get("/health").status_code == 200
                assert tracing._provider is not None
            assert tracing._provider is None
        assert "Tracing: ON" in caplog.text

    def test_lifespan_with_tracing_off(self, monkeypatch, caplog):
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
        monkeypatch.delenv("DISABLE_AUTH", raising=False)
        with caplog.at_level(logging.INFO, logger="api.tracing"):
            with TestClient(app) as test_client:
                assert test_client.get("/health").status_code == 200
        assert "Tracing: OFF" in caplog.text
        assert tracing._provider is None


# ── AC5: attribute value cap ────────────────────────────────────────────────


def _long_attribute_length(provider) -> int:
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("probe").start_as_current_span("s") as span:
        span.set_attribute("big", "x" * 20_000)
    (finished,) = exporter.get_finished_spans()
    return len(finished.attributes["big"])


class TestAttributeLengthLimit:
    def test_defaults_to_8192_when_unset(self, otlp_endpoint):
        assert "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT" not in os.environ
        provider = setup_tracing()
        assert _long_attribute_length(provider) == 8192
        # The default is applied via SpanLimits, not leaked into the environment.
        assert "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT" not in os.environ

    def test_explicit_limit_is_respected(self, otlp_endpoint, monkeypatch):
        monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "100")
        provider = setup_tracing()
        assert _long_attribute_length(provider) == 100

    def test_capture_spans_applies_the_same_cap(self):
        with capture_spans() as captured:
            with get_tracer().start_as_current_span("s") as span:
                span.set_attribute("big", "x" * 20_000)
        (finished,) = captured.spans
        assert len(finished.attributes["big"]) == 8192


# ── capture_spans ───────────────────────────────────────────────────────────


class TestCaptureSpans:
    def test_restores_the_setup_provider_on_exit(self, otlp_endpoint):
        provider = setup_tracing()
        with capture_spans() as captured:
            assert tracing._capture_provider is not provider
            with get_tracer().start_as_current_span("inside"):
                pass
        assert [s.name for s in captured.spans] == ["inside"]
        # Instrumentation is back on the setup provider, and the capture is gone.
        assert tracing._capture_provider is None
        assert LangChainInstrumentor().is_instrumented_by_opentelemetry
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        RunnableLambda(lambda x: x, name="after").invoke(1)
        assert any(s.name == "after" for s in exporter.get_finished_spans())
        assert "after" not in [s.name for s in captured.spans]

    def test_uninstruments_on_exit_when_nothing_was_installed(self):
        with capture_spans():
            assert LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
        assert get_tracer().__class__ is trace.NoOpTracer


# ── Driving a chat query through the real WS handler + graph nodes ──────────


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


_USAGE_1 = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
_USAGE_2 = {"input_tokens": 23, "output_tokens": 5, "total_tokens": 28}


def _scripted_responses() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "call_1"}],
            usage_metadata=_USAGE_1,
        ),
        AIMessage(content="The total is 5.", usage_metadata=_USAGE_2),
    ]


class _ScriptedChatModel(FakeMessagesListChatModel):
    """Fake chat model that tolerates `bind_tools` (as `create_react_agent` needs)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _build_agent(llm) -> PlanningAgent:
    """The real PlanningAgent + the real ReAct node, minus the router/planner LLM steps."""
    react = create_react_agent(llm, [add_numbers])
    workflow = StateGraph(AnalysisState)
    workflow.add_node("react_agent", create_react_node(react))
    workflow.set_entry_point("react_agent")
    workflow.add_edge("react_agent", END)
    return PlanningAgent(workflow.compile(), "AAPL")


def _wire_chat(monkeypatch, llm=None):
    llm = llm or _ScriptedChatModel(responses=_scripted_responses())
    monkeypatch.setattr(chat, "create_llm_pair", lambda *a, **k: (llm, llm))
    monkeypatch.setattr(
        "agents.graph.analyst_graph.create_sec_qa_agent",
        lambda ticker, llm, **k: _build_agent(llm),
    )


def _auth(**extra) -> dict:
    return {
        "type": "auth",
        "google_api_key": "byok-key",
        "openai_api_key": "byok-key",
        "anthropic_api_key": "byok-key",
        **extra,
    }


def _ask(client, auth: dict, query: str = "what is 2 + 3?") -> dict:
    with client.websocket_connect("/ws/chat/AAPL") as ws:
        ws.send_text(json.dumps(auth))
        _drain_until(ws, "auth_success")
        ws.send_text(json.dumps({"type": "query", "message": query}))
        return _drain_until(ws, "response")


def _by_id(spans):
    return {s.context.span_id: s for s in spans}


def _root_of(span, by_id):
    while span.parent is not None:
        span = by_id[span.parent.span_id]
    return span


def _attr_strings(spans):
    """Every attribute value (and span-event attribute value) as a string."""
    for s in spans:
        for v in s.attributes.values():
            yield str(v)
        for event in s.events:
            for v in event.attributes.values():
                yield str(v)


class TestChatQueryTrace:
    # AC3
    def test_one_query_is_one_trace_rooted_at_chat_query(self, client, temp_db, monkeypatch):
        _wire_chat(monkeypatch)
        with capture_spans() as captured:
            response = _ask(client, _auth())
        assert response["message"] == "The total is 5."

        spans = captured.spans
        by_id = _by_id(spans)
        roots = [s for s in spans if s.parent is None]
        assert [r.name for r in roots] == ["chat.query"]
        assert len({s.context.trace_id for s in spans}) == 1

        kinds = {s.attributes.get(KIND) for s in spans}
        assert {LLM_KIND, TOOL_KIND, CHAIN_KIND} <= kinds
        assert "react_agent" in {s.name for s in spans}  # the graph step
        assert "add_numbers" in {s.name for s in spans if s.attributes.get(KIND) == TOOL_KIND}
        for s in spans:  # every span is a descendant of chat.query
            assert _root_of(s, by_id) is roots[0]
        assert sum(1 for s in spans if s.attributes.get(KIND) == LLM_KIND) == 2

    def test_chat_query_attributes_anonymous(self, client, temp_db, monkeypatch):
        _wire_chat(monkeypatch)
        with capture_spans() as captured:
            _ask(client, _auth())
        (root,) = [s for s in captured.spans if s.name == "chat.query"]
        assert root.attributes["ticker"] == "AAPL"
        assert root.attributes["free_trial"] is False
        assert root.attributes["model_id"]
        assert "session_id" not in root.attributes

    def test_chat_query_attributes_signed_in_carry_session_id(self, client, temp_db, monkeypatch):
        _wire_chat(monkeypatch)
        with capture_spans() as captured:
            with client.websocket_connect("/ws/chat/AAPL") as ws:
                ws.send_text(json.dumps(_auth(user_id="user_sentinelUSER123")))
                session_id = _drain_until(ws, "auth_success")["session_id"]
                ws.send_text(json.dumps({"type": "query", "message": "what is 2 + 3?"}))
                _drain_until(ws, "response")
        (root,) = [s for s in captured.spans if s.name == "chat.query"]
        assert session_id
        assert root.attributes["session_id"] == session_id

    # AC4
    def test_llm_spans_carry_token_counts(self, client, temp_db, monkeypatch):
        _wire_chat(monkeypatch)
        with capture_spans() as captured:
            _ask(client, _auth())
        llm_spans = sorted(
            (s for s in captured.spans if s.attributes.get(KIND) == LLM_KIND),
            key=lambda s: s.start_time,
        )
        counts = [
            (
                s.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT],
                s.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION],
                s.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL],
            )
            for s in llm_spans
        ]
        assert counts == [(11, 7, 18), (23, 5, 28)]
        for s in llm_spans:
            assert s.end_time > s.start_time  # timings are recorded

    # Error paths: recorded on the span, behaviour unchanged for the client.
    @pytest.mark.parametrize(
        "exc, client_message",
        [
            (LLMTimeoutError("LLM call 'x' timed out"), "The model took too long to respond."),
            (RuntimeError("boom"), "Error processing your query."),
        ],
    )
    def test_agent_errors_are_recorded_on_chat_query(
        self, client, temp_db, monkeypatch, exc, client_message
    ):
        class _FailingAgent:
            def stream(self, *a, **k):
                async def _gen():
                    raise exc
                    yield  # pragma: no cover
                return _gen()

        monkeypatch.setattr(chat, "create_llm_pair", lambda *a, **k: (None, None))
        monkeypatch.setattr(
            "agents.graph.analyst_graph.create_sec_qa_agent", lambda *a, **k: _FailingAgent()
        )
        with capture_spans() as captured:
            with client.websocket_connect("/ws/chat/AAPL") as ws:
                ws.send_text(json.dumps(_auth()))
                _drain_until(ws, "auth_success")
                ws.send_text(json.dumps({"type": "query", "message": "hi"}))
                error = _drain_until(ws, "error")
        assert client_message in error["message"]

        (root,) = [s for s in captured.spans if s.name == "chat.query"]
        assert root.status.status_code == StatusCode.ERROR
        (event,) = [e for e in root.events if e.name == "exception"]
        assert event.attributes["exception.type"].endswith(type(exc).__name__)


# ── AC6 and identity: secrets never reach exported spans ────────────────────

_KEY_SENTINEL = "sk-SENTINEL-provider-key-0123456789"
_USER_SENTINEL = "user_SENTINELuserid0123456789"


class TestNoSecretsInSpans:
    def _run_with_real_provider_model(self, client, monkeypatch, auth):
        """A real ChatOpenAI holding the sentinel key; only its HTTP call is stubbed."""
        script = iter(_scripted_responses())

        def _next(*a, **k):
            return ChatResult(generations=[ChatGeneration(message=next(script))])

        async def _anext(*a, **k):
            return _next()

        monkeypatch.setattr(ChatOpenAI, "_generate", _next)
        monkeypatch.setattr(ChatOpenAI, "_agenerate", _anext)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=_KEY_SENTINEL)
        _wire_chat(monkeypatch, llm=llm)
        with capture_spans() as captured:
            _ask(client, auth)
        return captured.spans

    def test_provider_api_key_never_appears_in_span_attributes(self, client, temp_db, monkeypatch):
        auth = _auth(google_api_key=_KEY_SENTINEL, openai_api_key=_KEY_SENTINEL,
                     anthropic_api_key=_KEY_SENTINEL)
        spans = self._run_with_real_provider_model(client, monkeypatch, auth)

        assert any(s.attributes.get(KIND) == LLM_KIND for s in spans)
        leaked = [v for v in _attr_strings(spans) if _KEY_SENTINEL in v]
        assert leaked == []

    def test_raw_user_id_never_appears_in_span_attributes(self, client, temp_db, monkeypatch):
        _wire_chat(monkeypatch)
        with capture_spans() as captured:
            _ask(client, _auth(user_id=_USER_SENTINEL))

        spans = captured.spans
        assert any(s.attributes.get(KIND) == LLM_KIND for s in spans)
        leaked = [v for v in _attr_strings(spans) if _USER_SENTINEL in v]
        assert leaked == []
