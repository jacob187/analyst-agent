"""``run``: ask every question through one architecture and record metrics."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from api.tracing import capture_spans, get_tracer
from evals.benchmark.architectures import Builder, get_builder
from evals.benchmark.common import (
    DEFAULT_TIMEOUT_S,
    BenchmarkError,
    provider_api_key,
    resolve_model,
    scrub_secrets,
)
from evals.benchmark.isolation import DB_ISOLATED, isolated_db
from evals.benchmark.metrics import ROOT_SPAN_NAME, SPAN_ATTRIBUTE_CAP, SpanMetrics, extract_metrics
from evals.benchmark.pricing import pricing_from_model
from evals.benchmark.questions import Question, question_set_hash
from evals.benchmark.runs import QUESTIONS_DIR, RUN_FILE, write_model
from evals.benchmark.schemas import QuestionResult, RunMeta
from agents.model_registry import ModelDef

_ERROR_TEXT_LIMIT = 500
_REPO_ROOT = Path(__file__).resolve().parents[2]

_RUN_NOTES = [
    "Questions run sequentially in one process against a fresh temporary SQLite database "
    "created for this run. Filing analyses are cached in that database, so later questions "
    "on the same filing can be cheaper than the first, but nothing carries over between runs.",
    "Token counts come from LLM spans; calls whose span has no usage are counted in "
    "llm_calls_without_usage and make that question's cost unknown.",
    "Each question records its tool calls' inputs and outputs as judge evidence (outermost tool "
    "spans only, internal tools such as load_skill excluded), bounded per output and per question; "
    "evidence_truncated marks questions where anything was cut.",
]


def preflight(graph: str, model_id: str | None) -> tuple[Builder, ModelDef]:
    """Validate flags and credentials before spending anything.

    Raises BenchmarkError for an unknown architecture or model, a registry
    without a free-tier model (when --model is omitted), or a missing key.
    """
    builder = get_builder(graph)
    model = resolve_model(model_id)
    provider_api_key(model)  # raises if the provider key is not in the environment
    return builder, model


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> tuple[str, bool | None]:
    sha = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    return sha, (None if status is None else bool(status))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def _collect_answer(agent: Any, question: Question) -> str:
    answer: str | None = None
    async for event in agent.stream({"messages": [HumanMessage(content=question.question)]}):
        if event.get("type") == "response":
            message = event.get("message")
            answer = message if isinstance(message, str) else str(message)
    if not answer or not answer.strip():
        raise RuntimeError("agent stream ended without a response event")
    return answer


def _describe(exc: BaseException) -> str:
    text = str(exc) or "(no message)"
    return scrub_secrets(f"{type(exc).__name__}: {text}")[:_ERROR_TEXT_LIMIT]


async def run_question(builder: Builder, question: Question, model_id: str, timeout_s: float) -> QuestionResult:
    """Run one question; never raises (failures become ``status="failed"``)."""
    answer: str | None = None
    failure: str | None = None
    metrics = SpanMetrics()

    # A larger attribute cap than the app's 8192 so tool outputs kept as judge
    # evidence are not cut mid-table; this provider is the benchmark's own.
    with capture_spans(max_attribute_length=SPAN_ATTRIBUTE_CAP) as captured:
        try:
            agent = await builder(question.ticker, model_id)
            with get_tracer().start_as_current_span(
                ROOT_SPAN_NAME,
                attributes={"question_id": question.id, "ticker": question.ticker},
            ):
                answer = await asyncio.wait_for(_collect_answer(agent, question), timeout_s)
        except asyncio.TimeoutError:
            failure = f"TimeoutError: no answer within {timeout_s:g}s"
        except Exception as e:  # one question failing must not abort the run
            failure = _describe(e)
        metrics = extract_metrics(captured.spans)

    errors = [scrub_secrets(e) for e in metrics.errors]
    if failure and failure not in errors:
        errors.insert(0, failure)
    return QuestionResult(
        id=question.id,
        ticker=question.ticker,
        category=question.category,
        question=question.question,
        status="failed" if failure else "ok",
        answer=None if failure else answer,
        llm_calls=metrics.llm_calls,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        total_tokens=metrics.total_tokens,
        llm_calls_without_usage=metrics.llm_calls_without_usage,
        spans_outside_root_trace=metrics.spans_outside_root_trace,
        latency_s=round(metrics.latency_s, 3),
        llm_time_s=round(metrics.llm_time_s, 3),
        tool_calls=metrics.tool_calls,
        errors=errors,
        evidence=[
            e.model_copy(update={"input": scrub_secrets(e.input), "output": scrub_secrets(e.output)})
            for e in metrics.evidence
        ],
        evidence_truncated=metrics.evidence_truncated,
    )


async def run_benchmark(
    graph: str,
    model: ModelDef,
    questions: list[Question],
    out_dir: Path | str,
    *,
    only: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    progress: Callable[[str], None] = print,
) -> RunMeta:
    """Run ``questions`` (optionally just ``only`` ids) and write ``out_dir``.

    ``out_dir/run.json`` is written up front and finalized at the end; each
    question's JSON lands as soon as it finishes.
    """
    out_dir = Path(out_dir)
    if (out_dir / RUN_FILE).exists():
        raise BenchmarkError(f"{out_dir} already holds a run; choose a new --out directory.")

    selected = questions
    if only:
        known = {q.id for q in questions}
        unknown = [i for i in only if i not in known]
        if unknown:
            raise BenchmarkError(f"Unknown question id(s) in --only: {', '.join(unknown)}")
        selected = [q for q in questions if q.id in set(only)]

    builder = get_builder(graph)
    # Entered before run.json is written: a failure to set up the database must
    # not leave behind a run directory that blocks the re-run.
    async with isolated_db():
        return await _run_selected(graph, builder, model, questions, selected, out_dir, timeout_s, progress)


async def _run_selected(
    graph: str,
    builder: Builder,
    model: ModelDef,
    questions: list[Question],
    selected: list[Question],
    out_dir: Path,
    timeout_s: float,
    progress: Callable[[str], None],
) -> RunMeta:
    sha, dirty = git_state()
    meta = RunMeta(
        run_id=uuid.uuid4().hex,
        architecture=graph,
        model_id=model.id,
        git_sha=sha,
        git_dirty=dirty,
        date=_now(),
        question_set_hash=question_set_hash(questions),
        question_ids=[q.id for q in selected],
        tavily_enabled=bool((os.getenv("TAVILY_API_KEY") or "").strip()),
        pricing=pricing_from_model(model),
        db=DB_ISOLATED,
        notes=list(_RUN_NOTES),
    )
    write_model(out_dir / RUN_FILE, meta)

    failed = 0
    for index, question in enumerate(selected, start=1):
        result = await run_question(builder, question, model.id, timeout_s)
        write_model(out_dir / QUESTIONS_DIR / f"{question.id}.json", result)
        failed += result.status == "failed"
        progress(
            f"[{index}/{len(selected)}] {question.id}: {result.status} "
            f"{result.latency_s:.1f}s, {result.llm_calls} LLM calls, {result.total_tokens} tokens"
            + (f" ({result.errors[0]})" if result.errors else "")
        )

    meta.finished_at = _now()
    write_model(out_dir / RUN_FILE, meta)
    progress(f"Done: {len(selected) - failed}/{len(selected)} answered. Wrote {out_dir}")
    return meta
