"""AC2 (span -> metrics) and AC7 (cost from price fields; missing is unknown).

Spans are built with the real OpenTelemetry SDK and its in-memory exporter, no mocks.
"""

import time

import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from agents.model_registry import get_all_models
from evals.benchmark.metrics import ROOT_SPAN_NAME, extract_metrics
from evals.benchmark.pricing import estimate_cost, pricing_from_model
from evals.benchmark.schemas import Pricing

pytestmark = pytest.mark.eval_unit

KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND


def _llm(prompt=None, completion=None, total=None) -> dict:
    attrs = {KIND: "LLM"}
    for key, value in (
        (SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt),
        (SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion),
        (SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total),
    ):
        if value is not None:
            attrs[key] = value
    return attrs


def _new_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _record_question() -> list:
    tracer, exporter = _new_tracer()
    with tracer.start_as_current_span(ROOT_SPAN_NAME):
        with tracer.start_as_current_span("ChatModel", attributes=_llm(100, 10, 110)):
            time.sleep(0.002)
        # Tool B starts after tool A but A is the parent, so B finishes first.
        with tracer.start_as_current_span("tool_a_span", attributes={KIND: "TOOL", "tool.name": "get_risk_factors"}):
            time.sleep(0.002)
            with tracer.start_as_current_span("get_stock_price", attributes={KIND: "TOOL"}) as failing:
                failing.set_status(Status(StatusCode.ERROR, "yfinance timeout"))
        with tracer.start_as_current_span("ChatModel", attributes=_llm(prompt=300, completion=40)):  # no total
            pass
        with tracer.start_as_current_span("ChatModel", attributes=_llm()):  # no usage at all
            pass
        with tracer.start_as_current_span("router", attributes={KIND: "CHAIN"}):
            pass
    return exporter.get_finished_spans()


class TestExtractMetrics:
    def test_counts_tokens_calls_and_latency(self):
        spans = _record_question()
        m = extract_metrics(spans)

        assert m.llm_calls == 3
        assert m.input_tokens == 400
        assert m.output_tokens == 50
        assert m.total_tokens == 110 + 340  # a missing total is prompt + completion
        assert m.llm_calls_without_usage == 1
        (root,) = [s for s in spans if s.name == ROOT_SPAN_NAME]
        assert m.latency_s == pytest.approx((root.end_time - root.start_time) / 1e9)
        assert m.latency_s > 0.003
        assert 0 < m.llm_time_s < m.latency_s

    def test_tool_calls_are_ordered_by_start_with_names_and_errors(self):
        m = extract_metrics(_record_question())
        # tool.name attribute wins; the span name is the fallback.
        assert [t.name for t in m.tool_calls] == ["get_risk_factors", "get_stock_price"]
        assert [t.error for t in m.tool_calls] == [False, True]
        assert all(t.duration_s >= 0 for t in m.tool_calls)

    def test_errors_come_from_error_spans(self):
        m = extract_metrics(_record_question())
        assert m.errors == ["get_stock_price: yfinance timeout"]

    def test_exception_event_is_used_when_status_has_no_description(self):
        tracer, exporter = _new_tracer()
        with tracer.start_as_current_span(ROOT_SPAN_NAME):
            try:
                with tracer.start_as_current_span("tool_x", attributes={KIND: "TOOL"}):
                    raise ValueError("bad ticker")
            except ValueError:
                pass
        m = extract_metrics(exporter.get_finished_spans())
        assert m.errors and "tool_x" in m.errors[0]
        assert "bad ticker" in m.errors[0]

    def test_root_error_is_not_double_reported(self):
        tracer, exporter = _new_tracer()
        with pytest.raises(RuntimeError):
            with tracer.start_as_current_span(ROOT_SPAN_NAME):
                raise RuntimeError("boom")
        assert extract_metrics(exporter.get_finished_spans()).errors == []

    def test_spans_in_another_trace_still_count_and_are_flagged(self):
        # A worker thread that does not copy contextvars starts its own trace.
        tracer, exporter = _new_tracer()
        with tracer.start_as_current_span("orphan-chain"):
            with tracer.start_as_current_span("ChatModel", attributes=_llm(999, 1, 1000)):
                pass
            with tracer.start_as_current_span("orphan_tool", attributes={KIND: "TOOL"}):
                pass
        with tracer.start_as_current_span(ROOT_SPAN_NAME):
            with tracer.start_as_current_span("ChatModel", attributes=_llm(1, 2, 3)):
                pass
        m = extract_metrics(exporter.get_finished_spans())
        assert (m.llm_calls, m.input_tokens, m.output_tokens, m.total_tokens) == (2, 1000, 3, 1003)
        assert [t.name for t in m.tool_calls] == ["orphan_tool"]
        assert m.spans_outside_root_trace == 2

    def test_nothing_is_outside_the_root_trace_normally(self):
        assert extract_metrics(_record_question()).spans_outside_root_trace == 0

    def test_no_spans_is_all_zero(self):
        m = extract_metrics([])
        assert (m.llm_calls, m.total_tokens, m.latency_s, m.tool_calls, m.errors) == (0, 0, 0.0, [], [])


class TestCost:
    def test_cost_uses_both_prices_per_million_tokens(self):
        pricing = Pricing(input_price_per_mtok=2.0, output_price_per_mtok=10.0, prices_as_of="2026-01-15")
        assert estimate_cost(pricing, 1_000_000, 100_000) == pytest.approx(3.0)

    def test_zero_price_is_a_real_price_not_unknown(self):
        pricing = Pricing(input_price_per_mtok=0.0, output_price_per_mtok=0.0)
        assert estimate_cost(pricing, 5000, 500) == 0.0

    @pytest.mark.parametrize(
        "pricing",
        [
            None,
            Pricing(),
            Pricing(input_price_per_mtok=1.0),
            Pricing(output_price_per_mtok=1.0),
        ],
    )
    def test_missing_price_is_unknown_not_zero(self, pricing):
        assert estimate_cost(pricing, 5000, 500) is None

    def test_unobserved_token_usage_makes_cost_unknown(self):
        pricing = Pricing(input_price_per_mtok=1.0, output_price_per_mtok=1.0)
        assert estimate_cost(pricing, 5000, 500, llm_calls_without_usage=1) is None

    def test_pricing_is_read_from_the_registry_models_price_fields(self):
        for model in get_all_models():
            assert pricing_from_model(model) == Pricing(
                input_price_per_mtok=model.input_price_per_mtok,
                output_price_per_mtok=model.output_price_per_mtok,
                prices_as_of=model.prices_as_of,
            )
