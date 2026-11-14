"""``judge``: blind pairwise comparison of two runs' answers by an LLM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.model_registry import ModelDef
from evals.benchmark.common import BenchmarkError, provider_api_key, scrub_secrets
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement, JudgedQuestion, RunRef, ToolEvidence

PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.yaml"
_ERROR_TEXT_LIMIT = 500

# Evidence shown to the judge, per side (characters of rendered tool calls, before
# the small per-call headers and separators). A side over budget is trimmed
# deterministically: every call keeps an equal share of the budget, calls shorter
# than their share keep all of their text and hand the surplus to longer ones, and
# each over-share call keeps its head. A call is never trimmed below MIN_ITEM_CHARS;
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


def load_judge_prompt(path: Path = PROMPT_PATH) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"system", "user"} <= data.keys():
        raise BenchmarkError(f"{path} must define 'system' and 'user' prompt templates.")
    return {"system": data["system"], "user": data["user"]}


def build_judge_llm(model: ModelDef) -> Any:
    """The judge chat model for ``model`` (the one place a real judge LLM is built).

    Thinking is switched off explicitly. The verdict comes back through
    ``with_structured_output``, which for Anthropic models is forced tool use
    (``tool_choice`` = the verdict tool). Anthropic does not allow forced tool use
    together with extended thinking: for the budget-thinking model
    langchain-anthropic drops the forced ``tool_choice`` (the verdict can come back
    as prose and fail to parse), and for the adaptive-thinking models it still sends
    it, which the API rejects. A non-thinking judge forces the verdict tool; the
    evidence-based reasoning goes in the ``rationale``/``evidence_issues`` fields.
    ``create_llm`` already defaults to no thinking; passing it explicitly keeps the
    judge safe if that default ever changes. Other providers are unaffected.
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


def _coerce(result: Any) -> JudgeVerdict:
    if isinstance(result, JudgeVerdict):
        return result
    if isinstance(result, dict):
        return JudgeVerdict.model_validate(result)
    raise ValueError(f"judge returned no parsable verdict ({type(result).__name__})")


def check_comparable(run_a: LoadedRun, run_b: LoadedRun) -> None:
    if run_a.meta.question_set_hash != run_b.meta.question_set_hash:
        raise BenchmarkError(
            "Runs used different question sets (question_set_hash differs); "
            "re-run both against the same questions.yaml before judging."
        )


async def judge_runs(
    run_a: LoadedRun,
    run_b: LoadedRun,
    judge_llm: Any,
    judge_model_id: str,
    seed: int,
    *,
    progress: Callable[[str], None] = print,
) -> Judgement:
    """Judge every question both runs have an answer for.

    ``judge_llm`` needs ``with_structured_output(schema)`` returning something
    with ``ainvoke(messages)`` (any LangChain chat model). Answers are shown as
    FIRST/SECOND only: no architecture, model id or run label reaches the judge.
    Each answer comes with the (budget-trimmed) tool outputs its run retrieved, and
    the prompt states today's date so current filings/prices are not mistaken for
    hallucinations.
    """
    check_comparable(run_a, run_b)
    prompts = load_judge_prompt()
    today = as_of_date(run_a, run_b).isoformat()
    structured = judge_llm.with_structured_output(JudgeVerdict)

    judged: list[JudgedQuestion] = []
    shared_ids = [qid for qid in run_a.meta.question_ids if qid in set(run_b.meta.question_ids)]
    for index, qid in enumerate(shared_ids, start=1):
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
            SystemMessage(content=prompts["system"]),
            HumanMessage(
                content=prompts["user"].format(
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
        try:
            verdict = _coerce(await structured.ainvoke(messages))
        except Exception as e:  # one bad judge call must not abort the rest
            reason = scrub_secrets(f"{type(e).__name__}: {e}")[:_ERROR_TEXT_LIMIT]
            judged.append(JudgedQuestion(**base, status="error", reason=reason, a_is_first=first_is_a))
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
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        question_set_hash=run_a.meta.question_set_hash,
        a=ref_of(run_a),
        b=ref_of(run_b),
        questions=judged,
        as_of_date=today,
        evidence_budget_chars=EVIDENCE_BUDGET_CHARS,
    )
