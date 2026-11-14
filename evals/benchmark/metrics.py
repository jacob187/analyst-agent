"""Per-question metrics from captured OpenTelemetry spans (OpenInference attributes).

Spans come from ``api.tracing.capture_spans()``. Every question runs under a
``benchmark.question`` root span, so its spans share one trace id.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, Field

from evals.benchmark.schemas import ToolCall, ToolEvidence

ROOT_SPAN_NAME = "benchmark.question"

_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_LLM = OpenInferenceSpanKindValues.LLM.value
_TOOL = OpenInferenceSpanKindValues.TOOL.value
_ERROR_TEXT_LIMIT = 500

# Evidence capture. The app caps span string attributes at 8192 characters, which
# cuts SEC tables mid-row; benchmark runs capture with this larger cap instead
# (the runner passes it to ``capture_spans``; the app's default is untouched).
# It is twice EVIDENCE_OUTPUT_CAP because the span holds the output as a JSON
# string inside a serialized ToolMessage: newlines and quotes are escaped, so
# text can grow by a third or more. This way the stored cut is always the
# evidence cap, not the span cap landing mid-escape.
SPAN_ATTRIBUTE_CAP = 64_000
# What is stored per question in ``questions/<id>.json`` (characters):
EVIDENCE_OUTPUT_CAP = 30_000  # one tool output
EVIDENCE_INPUT_CAP = 2_000  # one tool input
EVIDENCE_TOTAL_CAP = 120_000  # inputs + outputs of all of a question's tool calls

_INPUT = SpanAttributes.INPUT_VALUE
_OUTPUT = SpanAttributes.OUTPUT_VALUE
# The LangChain instrumentation serializes a tool's ToolMessage as
# {"type": "tool", "data": {"content": <text>, ...}}.
_TOOL_MESSAGE_PREFIX = re.compile(r'^\{"type": "tool", "data": \{"content": ')


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
    evidence: list[ToolEvidence] = Field(default_factory=list)
    evidence_truncated: bool = False


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


def _partial_json_string(body: str) -> str:
    """Decode the body of a JSON string literal that may have been cut off mid-way."""
    for cut in range(13):  # drop a dangling escape such as a lone backslash or half a surrogate pair
        candidate = body[: len(body) - cut] if cut else body
        try:
            return json.loads(f'"{candidate}"')
        except json.JSONDecodeError:
            continue
    return body


def tool_output_text(raw: str) -> tuple[str, bool]:
    """A tool span's ``output.value`` as plain text, and whether the span cap cut it.

    Unwraps the serialized ToolMessage to its ``content``. When the span cap cut
    the JSON mid-way, the content written so far is still recovered.
    """
    cut = len(raw) >= SPAN_ATTRIBUTE_CAP
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _TOOL_MESSAGE_PREFIX.match(raw)
        if match and raw[match.end() : match.end() + 1] == '"':
            body = raw[match.end() + 1 :]
            try:
                text, _ = json.decoder.scanstring(body, 0)  # content closed before the cut
                return text, False
            except json.JSONDecodeError:
                return _partial_json_string(body), True
        return raw, cut
    content = data.get("data", {}).get("content") if isinstance(data, dict) else None
    if isinstance(content, str):
        return content, cut
    if isinstance(content, list):  # provider content blocks
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p), cut
    return raw, cut


def _is_nested_tool(span: ReadableSpan, by_id: dict[int, ReadableSpan]) -> bool:
    parent = span.parent
    seen: set[int] = set()
    while parent is not None and parent.span_id not in seen:
        seen.add(parent.span_id)
        ancestor = by_id.get(parent.span_id)
        if ancestor is None:
            return False
        if (ancestor.attributes or {}).get(_KIND) == _TOOL:
            return True
        parent = ancestor.parent
    return False


def _tool_name(span: ReadableSpan) -> str:
    return str((span.attributes or {}).get(SpanAttributes.TOOL_NAME) or span.name)


def extract_evidence(
    tool_spans: list[ReadableSpan], all_spans: list[ReadableSpan]
) -> tuple[list[ToolEvidence], bool]:
    """Bounded tool inputs/outputs for the judge, in call order.

    * A TOOL span nested inside another TOOL span (``web_search`` ->
      ``tavily_search``) is skipped: the outermost span's output is the evidence.
    * One output is cut at ``EVIDENCE_OUTPUT_CAP``, one input at
      ``EVIDENCE_INPUT_CAP`` and a question's total at ``EVIDENCE_TOTAL_CAP``.
      Once the total is spent, later calls are still listed (empty, truncated).
    * The returned flag is true when anything was cut or dropped.
    """
    by_id = {s.context.span_id: s for s in all_spans}
    remaining = EVIDENCE_TOTAL_CAP
    any_cut = False
    items: list[ToolEvidence] = []
    for span in sorted(tool_spans, key=lambda s: s.start_time or 0):
        name = _tool_name(span)
        if _is_nested_tool(span, by_id):
            continue
        attrs = span.attributes or {}
        is_error = span.status.status_code == StatusCode.ERROR
        text_in = str(attrs.get(_INPUT) or "")
        raw_out = str(attrs.get(_OUTPUT) or "")
        text_out, cut = tool_output_text(raw_out) if raw_out else ("", False)
        if not text_out and is_error:
            text_out = _error_text(span)

        if len(text_in) > EVIDENCE_INPUT_CAP:
            text_in, cut = text_in[:EVIDENCE_INPUT_CAP], True
        if len(text_in) > remaining:
            text_in, cut = text_in[:remaining], True
        remaining -= len(text_in)
        if len(text_out) > EVIDENCE_OUTPUT_CAP:
            text_out, cut = text_out[:EVIDENCE_OUTPUT_CAP], True
        if len(text_out) > remaining:
            text_out, cut = text_out[:remaining], True
        remaining -= len(text_out)
        any_cut = any_cut or cut
        items.append(ToolEvidence(tool=name, input=text_in, output=text_out, error=is_error, truncated=cut))
    return items, any_cut


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
    metrics.evidence, metrics.evidence_truncated = extract_evidence(tool_spans, spans)
    for span in tool_spans:
        metrics.tool_calls.append(
            ToolCall(
                name=_tool_name(span),
                duration_s=_seconds(span),
                error=span.status.status_code == StatusCode.ERROR,
            )
        )
    return metrics
