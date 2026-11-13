"""OpenTelemetry tracing for agent runs (OpenInference LangChain instrumentation).

Off unless an OTLP endpoint is configured. When on, every chat question yields
one trace rooted at a ``chat.query`` span (opened in ``api/routes/chat.py``)
with the LangGraph steps, tool calls and LLM calls (timings, token counts) as
descendants, exported over OTLP/HTTP.

Configuration is the standard OpenTelemetry environment (see ``.env.example``):
``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``,
``OTEL_EXPORTER_OTLP_HEADERS``, ``OTEL_SERVICE_NAME``, ``OTEL_TRACES_SAMPLER``
(+ ``_ARG``), ``OTEL_SDK_DISABLED``, ``OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT``
(defaults to 8192 here). Prompt/response redaction is OpenInference's
``OPENINFERENCE_HIDE_INPUTS`` / ``OPENINFERENCE_HIDE_OUTPUTS``.

The exporter is always the HTTP one; ``OTEL_EXPORTER_OTLP_PROTOCOL`` is not
consulted, so the endpoint must speak OTLP/HTTP.

The provider is deliberately not installed as the process-global OTel
provider: instrumentation and ``get_tracer()`` are handed it explicitly, which
keeps setup idempotent and lets tests swap providers freely.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, Sampler

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "analyst-agent"
DEFAULT_ATTRIBUTE_VALUE_LENGTH_LIMIT = "8192"
_TRACER_NAME = "analyst-agent"

# Provider installed by setup_tracing() (None while tracing is off).
_provider: TracerProvider | None = None
# Provider installed by capture_spans(); takes precedence in get_tracer().
_capture_provider: TracerProvider | None = None


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() == "true"


def _build_provider(sampler: Sampler | None = None) -> TracerProvider:
    """Build a TracerProvider with the app's resource and attribute-size cap.

    ``sampler=None`` lets the SDK resolve ``OTEL_TRACES_SAMPLER``/``_ARG``.
    The SDK reads ``OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`` while building the
    provider, so the default must be in the environment before that happens.
    """
    if not (os.getenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT") or "").strip():
        os.environ["OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = DEFAULT_ATTRIBUTE_VALUE_LENGTH_LIMIT
    service_name = (os.getenv("OTEL_SERVICE_NAME") or "").strip() or DEFAULT_SERVICE_NAME
    return TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name}),
        sampler=sampler,
    )


def setup_tracing() -> TracerProvider | None:
    """Install tracing if an OTLP endpoint is configured; return the provider.

    Returns None (and logs ``Tracing: OFF``) when ``OTEL_SDK_DISABLED=true`` or
    neither ``OTEL_EXPORTER_OTLP_ENDPOINT`` nor
    ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` is set. Idempotent: a second call
    returns the existing provider.
    """
    global _provider
    if _provider is not None:
        return _provider

    has_endpoint = bool(
        (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        or (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or "").strip()
    )
    if _env_flag("OTEL_SDK_DISABLED") or not has_endpoint:
        logger.info("Tracing: OFF")
        return None

    provider = _build_provider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _provider = provider
    logger.info(
        "Tracing: ON service=%s sampler=%s",
        provider.resource.attributes.get(SERVICE_NAME),
        provider.sampler.get_description(),
    )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Uninstrument LangChain and flush/shut down ``provider`` (None is a no-op).

    Never raises: it runs on app exit, where a flush failure (collector down)
    must not mask the real shutdown.
    """
    global _provider
    if provider is None:
        return
    try:
        if provider is _provider:
            LangChainInstrumentor().uninstrument()
            _provider = None
        provider.shutdown()
    except Exception:
        logger.warning("Tracing shutdown failed", exc_info=True)


def get_tracer() -> trace.Tracer:
    """Tracer for app spans (``chat.query``); a no-op tracer while tracing is off.

    Resolved at call time, so it follows ``setup_tracing`` / ``capture_spans``.
    """
    provider = _capture_provider or _provider
    if provider is None:
        return trace.NoOpTracer()
    return provider.get_tracer(_TRACER_NAME)


class CapturedSpans:
    """Finished spans recorded by ``capture_spans()``."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter

    @property
    def spans(self) -> list[ReadableSpan]:
        """Finished spans so far, in completion order (children before parents)."""
        return list(self._exporter.get_finished_spans())

    def clear(self) -> None:
        self._exporter.clear()


@contextmanager
def capture_spans() -> Iterator[CapturedSpans]:
    """Record every span from app code and LangChain/LangGraph in memory.

    For tests and the benchmark (LLM-call counts, token usage and latency come
    from the spans' OpenInference attributes, e.g. ``openinference.span.kind``
    == ``LLM`` and ``llm.token_count.*``). Installs a TracerProvider with an
    ``InMemorySpanExporter`` + ``SimpleSpanProcessor`` (always sampled), points
    ``get_tracer()`` and the LangChain instrumentation at it, and yields a
    ``CapturedSpans``. On exit the previous instrumentation (e.g. the
    ``setup_tracing`` provider) is restored; the yielded object stays readable.
    """
    global _capture_provider
    exporter = InMemorySpanExporter()
    provider = _build_provider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = LangChainInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    previous_capture = _capture_provider
    instrumentor.instrument(tracer_provider=provider)
    _capture_provider = provider
    try:
        yield CapturedSpans(exporter)
    finally:
        _capture_provider = previous_capture
        instrumentor.uninstrument()
        if _provider is not None:
            instrumentor.instrument(tracer_provider=_provider)
        provider.shutdown()
