"""Per-question metrics from captured OpenTelemetry spans (OpenInference attributes).

Spans come from ``api.tracing.capture_spans()``. Every question runs under a
``benchmark.question`` root span, so its spans share one trace id.
"""

from __future__ import annotations

from typing import Iterable

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, Field

from evals.benchmark.schemas import ToolCall

ROOT_SPAN_NAME = "benchmark.question"

_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_LLM = OpenInferenceSpanKindValues.LLM.value
_TOOL = OpenInferenceSpanKindValues.TOOL.value
_ERROR_TEXT_LIMIT = 500


class SpanMetrics(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls_without_usage: int = 0
    spans_outside_root_trace: int = 0
    latency_s: float = 0.0
    llm_time_s: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _seconds(span: ReadableSpan) -> float:
    if span.start_time is None or span.end_time is None:
        return 0.0
    return max(0.0, (span.end_time - span.start_time) / 1e9)


def _error_text(span: ReadableSpan) -> str:
    text = span.status.description or ""
    if not text:
        for event in span.events:
            if event.name == "exception":
                attrs = event.attributes or {}
                text = f"{attrs.get('exception.type', '')}: {attrs.get('exception.message', '')}".strip(": ")
                break
    text = f"{span.name}: {text or 'error'}"
    return text[:_ERROR_TEXT_LIMIT]


def extract_metrics(spans: Iterable[ReadableSpan]) -> SpanMetrics:
    """LLM-call count, token usage, latency, ordered tool calls and span errors.

    * Pass the spans captured for ONE question. All of them are counted, including
      LLM/TOOL spans outside the ``benchmark.question`` root's trace: calls made on
      worker threads that do not copy contextvars (e.g. a bare ThreadPoolExecutor)
      start their own trace but still belong to the question. Those are tallied in
      ``spans_outside_root_trace`` so the gap stays visible.
    * Latency is the root span's duration; without a root, the spread from the
      first start to the last end.
    * An LLM span with none of the ``llm.token_count.*`` attributes is counted
      in ``llm_calls_without_usage`` and adds nothing to the token totals.
    """
    spans = list(spans)
    roots = [s for s in spans if s.name == ROOT_SPAN_NAME and s.parent is None]
    root = roots[-1] if roots else None
    if not spans:
        return SpanMetrics()

    metrics = SpanMetrics()
    if root is not None:
        metrics.latency_s = _seconds(root)
    else:
        starts = [s.start_time for s in spans if s.start_time is not None]
        ends = [s.end_time for s in spans if s.end_time is not None]
        if starts and ends:
            metrics.latency_s = max(0.0, (max(ends) - min(starts)) / 1e9)

    tool_spans: list[ReadableSpan] = []
    for span in spans:
        attrs = span.attributes or {}
        kind = attrs.get(_KIND)
        if root is not None and kind in (_LLM, _TOOL) and span.context.trace_id != root.context.trace_id:
            metrics.spans_outside_root_trace += 1
        if kind == _LLM:
            metrics.llm_calls += 1
            metrics.llm_time_s += _seconds(span)
            prompt = attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT)
            completion = attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
            total = attrs.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL)
            if prompt is None and completion is None and total is None:
                metrics.llm_calls_without_usage += 1
            else:
                metrics.input_tokens += int(prompt or 0)
                metrics.output_tokens += int(completion or 0)
                metrics.total_tokens += int(
                    total if total is not None else (prompt or 0) + (completion or 0)
                )
        elif kind == _TOOL:
            tool_spans.append(span)
        # The root span goes ERROR when the question itself raised; the runner
        # reports that exception directly, so only descendants are listed here.
        if span.status.status_code == StatusCode.ERROR and span is not root:
            text = _error_text(span)
            if text not in metrics.errors:
                metrics.errors.append(text)

    tool_spans.sort(key=lambda s: s.start_time or 0)
    for span in tool_spans:
        metrics.tool_calls.append(
            ToolCall(
                name=str((span.attributes or {}).get(SpanAttributes.TOOL_NAME) or span.name),
                duration_s=_seconds(span),
                error=span.status.status_code == StatusCode.ERROR,
            )
        )
    return metrics
