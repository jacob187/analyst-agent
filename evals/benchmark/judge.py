"""``judge``: blind pairwise comparison of two runs' answers by an LLM."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from evals.benchmark.common import BenchmarkError, scrub_secrets
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement, JudgedQuestion, RunRef

PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.yaml"
_ERROR_TEXT_LIMIT = 500


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
    rationale: str = Field(description="Short justification: two or three sentences.")


def load_judge_prompt(path: Path = PROMPT_PATH) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"system", "user"} <= data.keys():
        raise BenchmarkError(f"{path} must define 'system' and 'user' prompt templates.")
    return {"system": data["system"], "user": data["user"]}


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
    """
    check_comparable(run_a, run_b)
    prompts = load_judge_prompt()
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
        messages = [
            SystemMessage(content=prompts["system"]),
            HumanMessage(
                content=prompts["user"].format(
                    ticker=qa.ticker,
                    category=qa.category,
                    question=qa.question,
                    first_answer=first.answer,
                    second_answer=second.answer,
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

        judged.append(
            JudgedQuestion(
                **base,
                status="judged",
                a_is_first=first_is_a,
                rationale=verdict.rationale,
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
    )
