"""``judge``: blind pairwise comparison of two runs' answers by an LLM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.llm_factory import structured_llm
from agents.model_registry import ModelDef
from agents.prompts import BENCHMARK_JUDGE_SYSTEM_PROMPT, BENCHMARK_JUDGE_USER_PROMPT
from api.tracing import capture_spans
from evals.benchmark.common import BenchmarkError, describe_error, provider_api_key, utc_now
from evals.benchmark.isolation import quiet_attribute_truncation
from evals.benchmark.metrics import extract_metrics
from evals.benchmark.pricing import estimate_cost
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement, JudgedQuestion, Pricing, RunRef, ToolEvidence

# Evidence shown to the judge, per side, in characters of rendered tool calls: each
# call's ``[i] tool:`` / ``input:`` / ``output:`` header lines count against the
# budget; only the blank-line separators and the closing omitted-calls line come on
# top. A side over budget is trimmed deterministically: every call keeps an equal
# share of the budget, calls shorter than their share keep all of their text and
# hand the surplus to longer ones, and each over-share call keeps its head plus a
# cut marker. A call is never rendered shorter than MIN_ITEM_CHARS (headers and the
# cut marker included, so a call with a long input can show little or no output);
# when there are too many calls for that, the earliest ones are kept and the rest
# are counted in a closing line. At 40k characters (~10k tokens) per side a judge
# prompt stays well inside any registry model's context window.
EVIDENCE_BUDGET_CHARS = 40_000
MIN_ITEM_CHARS = 500
_NOT_RECORDED = "(Evidence was not recorded for this answer.)"
_NO_EVIDENCE = "(No tool outputs were retrieved for this answer.)"


class JudgeVerdict(BaseModel):
    """What the judge returns, in the shuffled first/second frame."""

    winner: Literal["first", "second", "tie"] = Field(
        description="Which answer is better overall; 'tie' only if genuinely equal."
    )
    first_has_unique_factual_error: bool = Field(
        description="The first answer has a clear factual error the second does not make."
    )
    second_has_unique_factual_error: bool = Field(
        description="The second answer has a clear factual error the first does not make."
    )
    evidence_issues: str = Field(
        default="",
        description=(
            "Short note on claims that contradict or lack support in the evidence, or on "
            "evidence that looks cut or missing. Empty if there is nothing to note."
        ),
    )
    rationale: str = Field(description="Short justification: two or three sentences.")


def build_judge_llm(model: ModelDef) -> Any:
    """The judge chat model for ``model`` (the one place a real judge LLM is built).

    Thinking is switched off explicitly. The verdict comes back through
    ``agents.llm_factory.structured_llm``: for Anthropic models that is native
    JSON-schema output (``output_config``, no forced ``tool_choice``), for the other
    providers their default structured output. Thinking is not needed for a verdict
    whose reasoning goes in the ``rationale``/``evidence_issues`` fields, and leaving
    it off keeps judging cheap and the request shape the same across judge models.
    ``create_llm`` already defaults to no thinking; passing it explicitly keeps the
    judge that way if that default ever changes.
    """
    from agents.llm_factory import ThinkingConfig, create_llm

    return create_llm(model.id, provider_api_key(model), thinking=ThinkingConfig(enabled=False))


@dataclass(frozen=True)
class RenderedEvidence:
    text: str
    calls: int | None  # None = evidence not recorded for this run
    trimmed: bool  # the judge's budget cut something


def _evidence_body(index: int, item: ToolEvidence) -> str:
    head = f"[{index}] tool: {item.tool}" + (" (failed)" if item.error else "")
    body = f"{head}\ninput: {item.input}\noutput:\n{item.output}"
    if item.truncated:
        body += "\n[output was cut when the run was recorded]"
    return body


def _fair_shares(lengths: list[int], budget: int) -> list[int]:
    """Max-min fair split of ``budget``: short items keep everything, long ones share the rest."""
    shares = [0] * len(lengths)
    remaining, left = budget, len(lengths)
    for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        shares[i] = min(lengths[i], remaining // left)
        remaining -= shares[i]
        left -= 1
    return shares


def render_evidence(items: list[ToolEvidence] | None) -> RenderedEvidence:
    """Tool calls as prompt text, held to EVIDENCE_BUDGET_CHARS (deterministic; see there)."""
    if items is None:
        return RenderedEvidence(_NOT_RECORDED, None, False)
    if not items:
        return RenderedEvidence(_NO_EVIDENCE, 0, False)

    kept = items[: max(1, EVIDENCE_BUDGET_CHARS // MIN_ITEM_CHARS)]
    omitted = len(items) - len(kept)
    bodies = [_evidence_body(i, item) for i, item in enumerate(kept, start=1)]
    shares = _fair_shares([len(b) for b in bodies], EVIDENCE_BUDGET_CHARS)
    parts: list[str] = []
    trimmed = omitted > 0
    for body, share in zip(bodies, shares):
        if len(body) > share:
            trimmed = True
            marker = f"\n[... {len(body) - share} more characters cut by the judge's evidence budget]"
            body = body[: max(0, share - len(marker))] + marker
        parts.append(body)
    if omitted:
        parts.append(f"[... {omitted} further tool call(s) omitted by the judge's evidence budget]")
    return RenderedEvidence("\n\n".join(parts), len(items), trimmed)


def as_of_date(run_a: LoadedRun, run_b: LoadedRun) -> date:
    """The date the judge is told is today: the later run's date (judge time if neither parses)."""
    days: list[date] = []
    for run in (run_a, run_b):
        try:
            days.append(datetime.fromisoformat(run.meta.date).date())
        except ValueError:
            continue
    return max(days) if days else datetime.now(timezone.utc).date()


def a_is_first(seed: int, question_id: str) -> bool:
    """Per-question A/B order: deterministic in (seed, question id), so it is replayable."""
    return random.Random(f"{seed}:{question_id}").random() < 0.5


def _unshuffle(verdict: JudgeVerdict, a_first: bool) -> dict[str, Any]:
    if verdict.winner == "tie":
        winner = "tie"
    elif (verdict.winner == "first") == a_first:
        winner = "a"
    else:
        winner = "b"
    first_err, second_err = verdict.first_has_unique_factual_error, verdict.second_has_unique_factual_error
    return {
        "winner": winner,
        "a_unique_factual_error": first_err if a_first else second_err,
        "b_unique_factual_error": second_err if a_first else first_err,
    }


def _unwrap_verdict(result: dict[str, Any]) -> dict[str, Any]:
    """Peel a ``{"JudgeVerdict": {...}}`` wrapper some models put around the verdict.

    Only a single-key dict is unwrapped: when the key is the schema name, or when
    its sole value is a dict carrying the verdict's own fields.
    """
    if len(result) != 1:
        return result
    ((key, inner),) = result.items()
    if not isinstance(inner, dict):
        return result
    if key == JudgeVerdict.__name__ or "winner" in inner:
        return inner
    return result


def _coerce(result: Any) -> JudgeVerdict:
    if isinstance(result, JudgeVerdict):
        return result
    if isinstance(result, dict):
        return JudgeVerdict.model_validate(_unwrap_verdict(result))
    raise ValueError(f"judge returned no parsable verdict ({type(result).__name__})")


def check_comparable(run_a: LoadedRun, run_b: LoadedRun) -> None:
    if run_a.meta.question_set_hash != run_b.meta.question_set_hash:
        raise BenchmarkError(
            "Runs used different question sets (question_set_hash differs); "
            "re-run both against the same questions.yaml before judging."
        )


@dataclass(frozen=True)
class JudgeUsage:
    """What one judge call spent, from the OpenInference LLM spans it produced.

    ``input_tokens``/``output_tokens`` are None when no LLM span reported any
    usage. ``calls_without_usage`` counts the LLM calls that reported none (a
    call that left no span at all counts as one), so ``> 0`` means the figures
    are missing or a lower bound.
    """

    input_tokens: int | None
    output_tokens: int | None
    calls_without_usage: int


def judge_usage(spans: Any) -> JudgeUsage:
    """Token usage of the LLM spans captured around one judge call (``extract_metrics``, as for runs).

    The chat model's own span carries ``llm.token_count.*`` from the response's
    ``usage_metadata`` even under ``with_structured_output`` (the parser runs
    after the model span closes), for Anthropic, Google and OpenAI chat models.
    """
    m = extract_metrics(spans)
    if m.llm_calls == 0:
        return JudgeUsage(None, None, 1)
    if m.llm_calls_without_usage >= m.llm_calls:
        return JudgeUsage(None, None, m.llm_calls_without_usage)
    return JudgeUsage(m.input_tokens, m.output_tokens, m.llm_calls_without_usage)


async def _ask_judge(structured: Any, messages: list[Any]) -> tuple[JudgeVerdict | None, Exception | None, JudgeUsage]:
    """One judge call under ``capture_spans``: the verdict or the error, plus what it spent."""
    verdict: JudgeVerdict | None = None
    error: Exception | None = None
    # Only the token counts are read; the judge prompt (answers + evidence) is far past the span
    # attribute cap, so the SDK's per-attribute truncation warnings are silenced as in runs.
    with quiet_attribute_truncation(), capture_spans() as captured:
        try:
            verdict = _coerce(await structured.ainvoke(messages))
        except Exception as e:  # one bad judge call must not abort the rest
            error = e
    return verdict, error, judge_usage(captured.spans)


def errored_ids(judgement: Judgement) -> list[str]:
    """Questions whose judge call failed (the only ones a re-judge asks again)."""
    return [q.id for q in judgement.questions if q.status == "error"]


def rejudge_refusal(previous: Judgement, run_a: LoadedRun, run_b: LoadedRun, judge_model_id: str) -> str | None:
    """Why ``previous`` cannot be completed by re-judging its errors; None when it can.

    Only a judgement of these exact two runs by the same judge model is completed, so
    one judgement file never mixes judges or runs.
    """
    if (previous.a.run_id, previous.b.run_id) != (run_a.meta.run_id, run_b.meta.run_id):
        return "it judged other runs"
    if previous.judge_model != judge_model_id:
        return f"it was judged by {previous.judge_model}, not {judge_model_id}"
    return None


def _merge_spend(previous: Judgement | None, new_in: int | None, new_out: int | None, new_cost: float | None, new_without: int) -> dict[str, Any]:
    """The judgement's spend totals: this call's, plus ``previous``'s when it was completed.

    A completed judgement keeps the spend of its earlier calls, failed ones included:
    they were paid for. A part that is unknown makes the cost unknown, never 0.
    """
    if previous is None:
        return dict(judge_input_tokens=new_in, judge_output_tokens=new_out, judge_cost_usd=new_cost,
                    judge_calls_without_usage=new_without)

    def add(old: float | None, new: float | None) -> float | None:
        return None if old is None and new is None else (old or 0) + (new or 0)

    return dict(
        judge_input_tokens=add(previous.judge_input_tokens, new_in),
        judge_output_tokens=add(previous.judge_output_tokens, new_out),
        judge_cost_usd=None if previous.judge_cost_usd is None or new_cost is None else previous.judge_cost_usd + new_cost,
        judge_calls_without_usage=(previous.judge_calls_without_usage or 0) + new_without,
    )


async def judge_runs(
    run_a: LoadedRun,
    run_b: LoadedRun,
    judge_llm: Any,
    judge_model_id: str,
    seed: int,
    *,
    pricing: Pricing | None = None,
    progress: Callable[[str], None] = print,
    previous: Judgement | None = None,
) -> Judgement:
    """Judge every question both runs have an answer for.

    ``judge_llm`` needs ``with_structured_output(schema)`` returning something
    with ``ainvoke(messages)`` (any LangChain chat model). Answers are shown as
    FIRST/SECOND only: no architecture, model id or run label reaches the judge.
    Each answer comes with the (budget-trimmed) tool outputs its run retrieved, and
    the prompt states today's date so current filings/prices are not mistaken for
    hallucinations.

    The judge's own token usage is measured per question from the spans of its
    call and priced with ``pricing`` (the judge model's price snapshot); without
    prices, or without usage on a call, the cost is None (unknown), never 0.

    With ``previous`` (an earlier judgement of these runs by this judge model; see
    ``rejudge_refusal``) only its ``error`` questions are asked again. Every other
    entry is kept as it is, ``seed`` is replaced by the earlier seed (so the A/B order
    stays the recorded one) and the spend totals are the earlier totals plus this call's.
    """
    check_comparable(run_a, run_b)
    kept: dict[str, JudgedQuestion] = {}
    if previous is not None:
        if (why := rejudge_refusal(previous, run_a, run_b, judge_model_id)) is not None:
            raise BenchmarkError(f"Cannot complete the earlier judgement: {why}.")
        seed = previous.seed
        kept = {q.id: q for q in previous.questions if q.status != "error"}
        progress(f"Re-judging {len(errored_ids(previous))} errored question(s); keeping {len(kept)} earlier verdict(s)")
    today = as_of_date(run_a, run_b).isoformat()
    structured = structured_llm(judge_llm, JudgeVerdict)

    judged: list[JudgedQuestion] = []
    total_in = total_out = calls_without_usage = 0
    judge_calls = measured_calls = 0
    shared_ids = [qid for qid in run_a.meta.question_ids if qid in set(run_b.meta.question_ids)]
    for index, qid in enumerate(shared_ids, start=1):
        if qid in kept:
            judged.append(kept[qid])
            continue
        qa, qb = run_a.questions.get(qid), run_b.questions.get(qid)
        base = {"id": qid, "category": "", "ticker": ""}
        ref = qa or qb
        if ref is not None:
            base.update(category=ref.category, ticker=ref.ticker)

        missing = [
            name for name, q in (("A", qa), ("B", qb)) if q is None or not (q.answer or "").strip()
        ]
        if missing:
            judged.append(
                JudgedQuestion(
                    **base,
                    status="skipped",
                    reason=f"no answer in run {' and '.join(missing)}",
                )
            )
            progress(f"[{index}/{len(shared_ids)}] {qid}: skipped (no answer in run {' and '.join(missing)})")
            continue

        first_is_a = a_is_first(seed, qid)
        first, second = (qa, qb) if first_is_a else (qb, qa)
        first_ev, second_ev = render_evidence(first.evidence), render_evidence(second.evidence)
        messages = [
            SystemMessage(content=BENCHMARK_JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=BENCHMARK_JUDGE_USER_PROMPT.format(
                    ticker=qa.ticker,
                    category=qa.category,
                    question=qa.question,
                    today=today,
                    first_answer=first.answer,
                    first_evidence=first_ev.text,
                    second_answer=second.answer,
                    second_evidence=second_ev.text,
                )
            ),
        ]
        verdict, error, usage = await _ask_judge(structured, messages)
        judge_calls += 1
        measured_calls += usage.input_tokens is not None
        total_in += usage.input_tokens or 0
        total_out += usage.output_tokens or 0
        calls_without_usage += usage.calls_without_usage
        spent = dict(
            judge_input_tokens=usage.input_tokens,
            judge_output_tokens=usage.output_tokens,
            judge_cost_usd=(
                None
                if usage.input_tokens is None
                else estimate_cost(pricing, usage.input_tokens, usage.output_tokens or 0, usage.calls_without_usage)
            ),
        )
        if error is not None:
            reason = describe_error(error)
            judged.append(JudgedQuestion(**base, status="error", reason=reason, a_is_first=first_is_a, **spent))
            progress(f"[{index}/{len(shared_ids)}] {qid}: error ({reason})")
            continue

        a_ev, b_ev = (first_ev, second_ev) if first_is_a else (second_ev, first_ev)
        judged.append(
            JudgedQuestion(
                **base,
                status="judged",
                a_is_first=first_is_a,
                rationale=verdict.rationale,
                evidence_issues=verdict.evidence_issues.strip() or None,
                a_evidence_calls=a_ev.calls,
                b_evidence_calls=b_ev.calls,
                a_evidence_trimmed=a_ev.trimmed,
                b_evidence_trimmed=b_ev.trimmed,
                raw_verdict=verdict.model_dump(),
                **spent,
                **_unshuffle(verdict, first_is_a),
            )
        )
        progress(f"[{index}/{len(shared_ids)}] {qid}: judged")

    def ref_of(run: LoadedRun) -> RunRef:
        return RunRef(
            run_id=run.meta.run_id,
            dir=str(run.dir),
            architecture=run.meta.architecture,
            model_id=run.meta.model_id,
        )

    return Judgement(
        judge_model=judge_model_id,
        seed=seed,
        date=utc_now(),
        question_set_hash=run_a.meta.question_set_hash,
        a=ref_of(run_a),
        b=ref_of(run_b),
        questions=judged,
        as_of_date=today,
        evidence_budget_chars=EVIDENCE_BUDGET_CHARS,
        **_merge_spend(
            previous,
            # No call reported usage at all: the totals are unknown, not 0 (a partial set is a lower bound).
            None if judge_calls and not measured_calls else total_in,
            None if judge_calls and not measured_calls else total_out,
            estimate_cost(pricing, total_in, total_out, calls_without_usage),
            calls_without_usage,
        ),
        judge_prices_as_of=None if pricing is None else pricing.prices_as_of,
    )
