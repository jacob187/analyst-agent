"""``compare-judgements``: how well two judgements of the SAME run pair agree.

Purpose: decide whether a cheaper judge agrees well enough with an expensive one
to replace it. Both judgements must have judged the same two runs (same run ids);
a pair judged in the opposite A/B order is re-oriented to the first judgement's
order, so "A" always means the first judgement's run A. Only questions both judges
actually judged are compared; questions one of them skipped or errored on are
counted separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.benchmark.common import BenchmarkError, rate
from evals.benchmark.pricing import cost_text, usage_text
from evals.benchmark.schemas import Judgement, JudgedQuestion

AGREEMENT_THRESHOLD = 0.85
GUIDANCE = (
    f"Guidance, not a gate: agreement >= {AGREEMENT_THRESHOLD:.0%} on winners and on error flags "
    "-> the cheaper judge is acceptable."
)
RATIONALE_LIMIT = 200
_FLAGS = ("a_unique_factual_error", "b_unique_factual_error")
_WINNERS = ("a", "b", "tie")


def _clip(text: str | None) -> str:
    text = (text or "").strip()
    return text if len(text) <= RATIONALE_LIMIT else text[: RATIONALE_LIMIT - 3] + "..."


def cohens_kappa(first: list[str], second: list[str]) -> float | None:
    """Cohen's kappa over paired categorical labels; None when undefined (no pairs, or chance agreement is 1)."""
    n = len(first)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(first, second)) / n
    chance = sum((first.count(c) / n) * (second.count(c) / n) for c in _WINNERS)
    if chance >= 1:
        return None
    return round((observed - chance) / (1 - chance), 4)


def _flipped(q: JudgedQuestion) -> JudgedQuestion:
    """The same verdict seen with run A and run B exchanged."""
    return q.model_copy(
        update={
            "winner": {"a": "b", "b": "a", "tie": "tie", None: None}[q.winner],
            "a_unique_factual_error": q.b_unique_factual_error,
            "b_unique_factual_error": q.a_unique_factual_error,
        }
    )


def _orient(first: Judgement, second: Judgement) -> list[JudgedQuestion]:
    """``second``'s questions in ``first``'s run order; refuses judgements of different run pairs."""
    pair = (first.a.run_id, first.b.run_id)
    if (second.a.run_id, second.b.run_id) == pair:
        return list(second.questions)
    if (second.b.run_id, second.a.run_id) == pair:
        return [_flipped(q) for q in second.questions]
    raise BenchmarkError(
        "The two judgements are not of the same run pair: "
        f"first judged run ids {first.a.run_id} (A) vs {first.b.run_id} (B), "
        f"second judged {second.a.run_id} (A) vs {second.b.run_id} (B). "
        "Compare judgements of the same two runs."
    )


def _judge_block(j: Judgement) -> dict[str, Any]:
    return {
        "judge_model": j.judge_model,
        "seed": j.seed,
        "cost_usd": j.judge_cost_usd,
        "input_tokens": j.judge_input_tokens,
        "output_tokens": j.judge_output_tokens,
        "calls_without_usage": j.judge_calls_without_usage,
        "prices_as_of": j.judge_prices_as_of,
    }


def _side(q: JudgedQuestion) -> dict[str, Any]:
    return {
        "winner": q.winner,
        "a_unique_factual_error": bool(q.a_unique_factual_error),
        "b_unique_factual_error": bool(q.b_unique_factual_error),
        "rationale": _clip(q.rationale),
    }


def _stats(pairs: list[tuple[JudgedQuestion, JudgedQuestion]]) -> dict[str, Any]:
    n = len(pairs)
    stats: dict[str, Any] = {"compared": n}
    agreed = sum(x.winner == y.winner for x, y in pairs)
    stats["winner_agreed"] = agreed
    stats["winner_agreement"] = rate(agreed, n)
    for flag in _FLAGS:
        same = sum(bool(getattr(x, flag)) == bool(getattr(y, flag)) for x, y in pairs)
        stats[flag] = {
            "agreed": same,
            "agreement": rate(same, n),
            # Agreement is easy when flags are rare; these show how many flags there were to agree on.
            "both_flagged": sum(bool(getattr(x, flag)) and bool(getattr(y, flag)) for x, y in pairs),
            "either_flagged": sum(bool(getattr(x, flag)) or bool(getattr(y, flag)) for x, y in pairs),
        }
    return stats


def compare_judgements(first: Judgement, second: Judgement) -> dict[str, Any]:
    """Agreement report for two judgements of the same run pair (raises BenchmarkError otherwise)."""
    second_questions = {q.id: q for q in _orient(first, second)}
    first_questions = {q.id: q for q in first.questions}

    pairs: list[tuple[JudgedQuestion, JudgedQuestion]] = []
    only_one: list[str] = []
    for qid, x in first_questions.items():
        y = second_questions.get(qid)
        x_judged = x.status == "judged"
        y_judged = y is not None and y.status == "judged"
        if x_judged and y_judged:
            pairs.append((x, y))
        elif x_judged or y_judged:
            only_one.append(qid)
    only_one.extend(qid for qid, y in second_questions.items() if qid not in first_questions and y.status == "judged")

    by_category: dict[str, list[tuple[JudgedQuestion, JudgedQuestion]]] = defaultdict(list)
    disagreements: list[dict[str, Any]] = []
    for x, y in pairs:
        by_category[x.category].append((x, y))
        differs = (["winner"] if x.winner != y.winner else []) + [
            f for f in _FLAGS if bool(getattr(x, f)) != bool(getattr(y, f))
        ]
        if differs:
            disagreements.append(
                {"id": x.id, "category": x.category, "differs_on": differs, "first": _side(x), "second": _side(y)}
            )

    overall = _stats(pairs)
    rates = [overall["winner_agreement"], *(overall[f]["agreement"] for f in _FLAGS)]
    meets = None if any(r is None for r in rates) else all(r >= AGREEMENT_THRESHOLD for r in rates)
    return {
        "first_judge": _judge_block(first),
        "second_judge": _judge_block(second),
        "runs": {"a": first.a.model_dump(), "b": first.b.model_dump()},
        "question_set_hash": first.question_set_hash,
        "questions_in_first": len(first.questions),
        "questions_in_second": len(second.questions),
        "judged_by_only_one": sorted(only_one),
        **overall,
        "winner_kappa": cohens_kappa([x.winner for x, _ in pairs], [y.winner for _, y in pairs]),
        "by_category": {cat: _stats(ps) for cat, ps in sorted(by_category.items())},
        "disagreements": disagreements,
        "guidance": {"threshold": AGREEMENT_THRESHOLD, "text": GUIDANCE, "meets": meets},
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _judge_line(label: str, judge: dict[str, Any]) -> str:
    tokens = usage_text(judge["input_tokens"], judge["output_tokens"], judge["calls_without_usage"])
    return f"  {label}: {judge['judge_model']}, judge cost {cost_text(judge['cost_usd'])} ({tokens})"


def agreement_text(report: dict[str, Any]) -> str:
    """Short terminal digest, ending with the guidance rule and where this pair stands against it."""
    n = report["compared"]
    lines = [
        f"Compared {n} question(s) judged by both judges"
        + (f"; {len(report['judged_by_only_one'])} judged by only one" if report["judged_by_only_one"] else ""),
        _judge_line("first ", report["first_judge"]),
        _judge_line("second", report["second_judge"]),
        f"Winner agreement: {_pct(report['winner_agreement'])} ({report['winner_agreed']}/{n}); "
        f"Cohen's kappa: {'n/a' if report['winner_kappa'] is None else report['winner_kappa']}",
    ]
    for flag, name in zip(_FLAGS, ("run A", "run B")):
        f = report[flag]
        lines.append(
            f"Unique-factual-error flag agreement, {name}: {_pct(f['agreement'])} ({f['agreed']}/{n}; "
            f"flagged by both {f['both_flagged']}, by either {f['either_flagged']})"
        )
    if report["by_category"]:
        lines.append("By category (winner agreement):")
        for cat, s in report["by_category"].items():
            lines.append(f"  {cat}: {_pct(s['winner_agreement'])} ({s['winner_agreed']}/{s['compared']})")
    if report["disagreements"]:
        lines.append(f"Disagreements ({len(report['disagreements'])}):")
        for d in report["disagreements"]:
            lines.append(
                f"  {d['id']} [{d['category']}] differs on {', '.join(d['differs_on'])}: "
                f"first {d['first']['winner']} / second {d['second']['winner']}"
            )
            lines.append(f"    first:  {d['first']['rationale']}")
            lines.append(f"    second: {d['second']['rationale']}")
    lines.append(report["guidance"]["text"])
    meets = report["guidance"]["meets"]
    lines.append(
        "Observed: "
        + ("undefined (no questions judged by both)" if meets is None
           else "at or above the threshold on all three" if meets else "below the threshold on at least one")
    )
    return "\n".join(lines)
