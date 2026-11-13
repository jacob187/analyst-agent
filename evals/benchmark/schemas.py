"""On-disk shapes: ``run.json``, ``questions/<id>.json`` and ``judgement.json``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    duration_s: float
    error: bool = False


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
    latency_s: float = 0.0
    llm_time_s: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


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
