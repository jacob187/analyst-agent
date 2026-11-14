"""On-disk shapes: ``run.json``, ``questions/<id>.json`` and ``judgement.json``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    duration_s: float
    error: bool = False


class ToolEvidence(BaseModel):
    """One tool call's input and output, as the judge's evidence.

    ``output`` is the tool's text, unwrapped from the span's serialized message
    and bounded (see ``metrics`` for the caps); ``truncated`` is true when it
    (or the input) was cut, or dropped entirely because the question's total
    evidence budget was already spent.
    """

    tool: str
    input: str = ""
    output: str = ""
    error: bool = False
    truncated: bool = False


class Pricing(BaseModel):
    """Per-million-token prices snapshotted from the model registry at run time."""

    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    prices_as_of: str | None = None


class RunMeta(BaseModel):
    """``run.json``."""

    run_id: str
    architecture: str
    model_id: str
    git_sha: str
    git_dirty: bool | None = None
    date: str
    finished_at: str | None = None
    question_set_hash: str
    question_ids: list[str]
    tavily_enabled: bool
    pricing: Pricing
    # "isolated": the run used its own fresh SQLite database (no shared cached analyses).
    # None on runs recorded before isolation existed.
    db: str | None = None
    notes: list[str] = Field(default_factory=list)


class QuestionResult(BaseModel):
    """``questions/<id>.json``: one question's answer plus metrics from spans."""

    id: str
    ticker: str
    category: str
    question: str
    status: Literal["ok", "failed"]
    answer: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # LLM calls whose span carried no token counts. When > 0 the token totals
    # are a lower bound and the cost is reported as unknown.
    llm_calls_without_usage: int = 0
    # LLM/TOOL spans that ran outside the question's trace (thread pools that do
    # not copy contextvars). They are included in every count above.
    spans_outside_root_trace: int = 0
    latency_s: float = 0.0
    llm_time_s: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # Tool inputs/outputs the answer was built from (nested duplicate spans
    # collapsed to the outermost).
    # None = not recorded (files written before this field existed), which is
    # different from [] = the run made no evidence-bearing tool calls.
    evidence: list[ToolEvidence] | None = None
    # True when any evidence item was cut or dropped by the storage caps.
    evidence_truncated: bool = False


class JudgedQuestion(BaseModel):
    """One question's verdict, already un-shuffled back to run A / run B."""

    id: str
    category: str
    ticker: str
    status: Literal["judged", "skipped", "error"]
    reason: str | None = None
    # True when run A's answer was shown first (recorded shuffle).
    a_is_first: bool | None = None
    winner: Literal["a", "b", "tie"] | None = None
    a_unique_factual_error: bool | None = None
    b_unique_factual_error: bool | None = None
    rationale: str | None = None
    # The judge's note on claims that contradict or lack support in the evidence.
    evidence_issues: str | None = None
    # How many evidence items each run contributed (None = evidence not recorded
    # for that run) and whether the judge's per-side budget cut it. None on
    # judgements written before evidence was shown to the judge.
    a_evidence_calls: int | None = None
    b_evidence_calls: int | None = None
    a_evidence_trimmed: bool | None = None
    b_evidence_trimmed: bool | None = None
    # The verdict exactly as the judge returned it, in first/second terms.
    raw_verdict: dict | None = None


class RunRef(BaseModel):
    run_id: str
    dir: str
    architecture: str
    model_id: str


class Judgement(BaseModel):
    """``judgement.json``."""

    judge_model: str
    seed: int
    date: str
    question_set_hash: str
    a: RunRef
    b: RunRef
    questions: list[JudgedQuestion]
    # The date the judge was told is "today" (the later run's date) and the
    # per-side evidence character budget it was held to. None on older files.
    as_of_date: str | None = None
    evidence_budget_chars: int | None = None
