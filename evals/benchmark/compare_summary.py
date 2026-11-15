"""``compare-models`` summary: per-model metrics against a reference run, plus recommendations.

Every model's run is judged pairwise against the reference run (A = the model,
B = the reference). The per-model blocks reuse ``summary.build_summary`` for
means, costs and per-category aggregates and add the judge outcome on top.

Recommendation rules (also written verbatim into ``summary.json``):

* Win score and win+tie rate: the reference has no judgement against itself, so on
  those two it competes as a tie on every question (win score 0.5, win+tie rate
  100%). A challenger can only be recommended over it by beating that.
* Unique factual errors: the reference's are NOT assumed to be zero. They are read
  from the judgements, i.e. the reverse side of every candidate-vs-reference
  pairing (``reference_standing``). All pairings judge the same reference answers,
  so the same flawed answer shows up once per pairing; the standing therefore takes
  the **maximum** over pairings, never the sum. A pairing with no judged question
  is ignored. If no pairing was judged without judge errors, the count is unknown
  (``None``) and the reference cannot be shown to be clean. The reverse W/T/L per
  pairing is recorded for visibility only; it is not used to gate the reference,
  because it is measured against a different opponent in each pairing.
* ``win score`` = (wins + 0.5 * ties) / questions. A question the model failed to
  answer while the reference answered counts as a loss for the model.
* A model with any judge error in the relevant scope, no judgement, or an unknown
  cost per question is not a candidate; the reason is stated in the rationale.
* ``free_trial``: the cheapest candidate with win+tie rate >= ``MIN_WIN_TIE_RATE``
  and zero unique factual errors (for the reference: zero known unique errors in
  the pairings above).
* A model whose run answered no question at all (or whose run directory is missing
  or unreadable) is listed under ``excluded_models`` with the reason and a warning,
  and takes no part in the rules.
* ``chat_default``: the candidate with the highest win score among those costing at
  most ``chat_cost_multiple`` x the reference's mean cost per question.
* ``filing_analysis``: the ``chat_default`` rule on the ``filing_lookup`` category only.
* When no candidate remains the answer is ``model_id: null`` with an "insufficient
  data" rationale; nothing is guessed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from evals.benchmark.common import BenchmarkError
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement
from evals.benchmark.pricing import cost_text
from evals.benchmark.summary import build_summary, judge_usage_block, judgement_counts, sum_known

MIN_WIN_TIE_RATE = 0.85
DEFAULT_CHAT_COST_MULTIPLE = 3.0
FILING_CATEGORY = "filing_lookup"


def _rate(numerator: float, n: int) -> float | None:
    return None if n == 0 else round(numerator / n, 4)


def unusable_reason(run: LoadedRun) -> str | None:
    """Why ``run`` cannot be compared (no question has an answer); None when it can.

    A partly failed run is usable: its failed questions count as losses.
    """
    results = [run.questions[qid] for qid in run.meta.question_ids if qid in run.questions]
    if any((r.answer or "").strip() for r in results):
        return None
    if not results:
        return "the run recorded no question results"
    first_error = next((line for r in results for e in r.errors for line in e.splitlines()[:1] if line.strip()), None)
    detail = "" if first_error is None else f" (first error: {first_error.strip()[:200]})"
    return f"all {len(run.meta.question_ids)} questions failed{detail}"


def vs_reference(
    model_run: LoadedRun,
    reference_run: LoadedRun,
    judgement: Judgement | None,
    category: str | None = None,
) -> dict[str, Any] | None:
    """The model's judged record against the reference (None without a judgement)."""
    if judgement is None:
        return None
    entries = [q for q in judgement.questions if category is None or q.category == category]
    counts = judgement_counts(judgement.model_copy(update={"questions": entries}), "a")

    def answered(run: LoadedRun, qid: str) -> bool:
        result = run.questions.get(qid)
        return bool(result is not None and (result.answer or "").strip())

    skipped = [q for q in entries if q.status == "skipped"]
    unanswered = sum(answered(reference_run, q.id) and not answered(model_run, q.id) for q in skipped)
    n = counts["judged"] + unanswered
    return {
        "judged": counts["judged"],
        "wins": counts["wins"],
        "ties": counts["ties"],
        "losses": counts["losses"],
        "unanswered_by_model": unanswered,
        "other_skipped": len(skipped) - unanswered,
        "judge_errors": sum(q.status == "error" for q in entries),
        "unique_factual_errors": counts["unique_factual_errors"],
        "reference_unique_factual_errors": counts["opponent_unique_factual_errors"],
        "win_rate": _rate(counts["wins"], n),
        "win_or_tie_rate": _rate(counts["wins"] + counts["ties"], n),
        "win_score": _rate(counts["wins"] + 0.5 * counts["ties"], n),
    }


def _model_block(
    run: LoadedRun,
    reference_run: LoadedRun,
    run_block: dict[str, Any],
    judgement: Judgement | None,
    run_dir: str,
) -> dict[str, Any]:
    is_reference = run is reference_run
    by_category = {}
    for cat, agg in run_block["by_category"].items():
        by_category[cat] = {
            "n": agg["n"],
            "means": agg["means"],
            "cost_unknown_questions": agg["cost_unknown_questions"],
            "vs_reference": None if is_reference else vs_reference(run, reference_run, judgement, cat),
        }
    return {
        "model_id": run.meta.model_id,
        "is_reference": is_reference,
        "run_dir": run_dir,
        "run_id": run.meta.run_id,
        "questions": run_block["questions"],
        "vs_reference": None if is_reference else vs_reference(run, reference_run, judgement),
        # What judging this model against the reference cost, as recorded when it was judged
        # (a resumed judgement may have been made by another judge model than this invocation's).
        "judge": None if is_reference or judgement is None else judge_usage_block(judgement),
        "means": run_block["means"],
        "totals": run_block["totals"],
        "cost_unknown_questions": run_block["cost_unknown_questions"],
        "pricing": run_block["pricing"],
        "prices_as_of": run_block["pricing"]["prices_as_of"],
        "by_category": by_category,
    }


def reference_standing(pairings: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """The reference's own record, read from the reverse side of each pairing.

    ``pairings`` maps a challenger id to its ``vs_reference`` block (None = not judged).
    Reference wins are the challenger's losses plus the questions the challenger failed
    to answer; reference losses are the challenger's wins. ``unique_factual_errors`` is
    the maximum over pairings (the same reference answers are judged once per pairing, so
    summing would count one flawed answer N times); None when no pairing was judged
    without judge errors, since then a clean record is unproven.
    """
    used = {m: vs for m, vs in pairings.items() if vs is not None and vs["judged"] > 0}
    rows = [
        {
            "model_id": m,
            "wins": vs["losses"] + vs["unanswered_by_model"],
            "ties": vs["ties"],
            "losses": vs["wins"],
            "unique_factual_errors": vs["reference_unique_factual_errors"],
            "judge_errors": vs["judge_errors"],
        }
        for m, vs in used.items()
    ]
    verified = any(row["judge_errors"] == 0 for row in rows)
    errors = max((row["unique_factual_errors"] for row in rows), default=0)
    return {
        "pairings_used": len(rows),
        "unique_factual_errors": errors if rows and (verified or errors > 0) else None,
        "pairings": rows,
    }


def _attach_reference_standing(models: dict[str, dict[str, Any]], reference_id: str) -> None:
    ref = models[reference_id]
    others = [b for m, b in models.items() if m != reference_id]
    ref["reference_standing"] = reference_standing({b["model_id"]: b["vs_reference"] for b in others})
    for cat, agg in ref["by_category"].items():
        agg["reference_standing"] = reference_standing(
            {b["model_id"]: b["by_category"][cat]["vs_reference"] for b in others if cat in b["by_category"]}
        )


def _standing(is_reference: bool, vs: dict[str, Any] | None, scope: dict[str, Any]) -> dict[str, Any] | None:
    """The numbers the rules compare; None when there is no usable judgement.

    The reference is a tie with itself on win score and win+tie rate; its unique
    factual errors come from ``reference_standing`` (None = not shown to be clean).
    """
    if is_reference:
        errors = scope["reference_standing"]["unique_factual_errors"]
        return {"win_score": 0.5, "win_or_tie_rate": 1.0, "unique_factual_errors": errors, "judge_errors": 0}
    if vs is None or vs["win_score"] is None:
        return None
    return vs


def _candidates(
    models: dict[str, dict[str, Any]], category: str | None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Models the rules may pick from, and why each of the others was left out."""
    candidates: list[dict[str, Any]] = []
    excluded: list[str] = []
    for model_id, block in models.items():
        scope = block if category is None else block["by_category"].get(category)
        if scope is None:
            excluded.append(f"{model_id} (no {category} questions)")
            continue
        vs = scope["vs_reference"]
        standing = _standing(block["is_reference"], vs, scope)
        cost = scope["means"]["cost_usd"]
        if vs is not None and vs["judge_errors"]:
            excluded.append(f"{model_id} ({vs['judge_errors']} judge errors)")
        elif standing is None:
            excluded.append(f"{model_id} (no judged questions)")
        elif cost is None:
            excluded.append(f"{model_id} (unknown cost)")
        else:
            candidates.append({"model_id": model_id, "cost": cost, **standing})
    return candidates, excluded


def _insufficient(why: str, excluded: list[str]) -> dict[str, Any]:
    detail = f" Excluded: {', '.join(excluded)}." if excluded else ""
    return {"model_id": None, "rationale": f"Insufficient data: {why}.{detail}"}


def _note(excluded: list[str]) -> str:
    return f" (left out: {', '.join(excluded)})" if excluded else ""


def _reference_not_clean(models: dict[str, dict[str, Any]]) -> str | None:
    """Why the reference cannot be the free trial on unique errors (None when it is shown clean)."""
    for model_id, block in models.items():
        if block["is_reference"]:
            errors = block["reference_standing"]["unique_factual_errors"]
            if errors is None:
                return f"{model_id} (reference: no judged pairing, so its unique factual errors are unknown)"
            if errors:
                noun = "error" if errors == 1 else "errors"
                return f"{model_id} (reference: {errors} unique factual {noun} in the worst pairing)"
    return None


def _pick_free_trial(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates, excluded = _candidates(models, None)
    passing = [
        c for c in candidates
        if c["win_or_tie_rate"] >= MIN_WIN_TIE_RATE and c["unique_factual_errors"] == 0
    ]
    if (reason := _reference_not_clean(models)) is not None:
        excluded.append(reason)
    if not passing:
        why = (
            "no model with a known cost and a clean judgement reached "
            f"{MIN_WIN_TIE_RATE:.0%} win+tie with no unique factual errors"
            if candidates else "no model has both a known cost and a complete judgement"
        )
        return _insufficient(why, excluded)
    best = min(passing, key=lambda c: (c["cost"], -c["win_score"], c["model_id"]))
    return {
        "model_id": best["model_id"],
        "rationale": (
            f"Cheapest qualifying model: ${best['cost']:.4f}/question, win+tie "
            f"{best['win_or_tie_rate']:.0%}, no unique factual errors{_note(excluded)}."
        ),
    }


def _pick_best_within_budget(
    models: dict[str, dict[str, Any]], reference_id: str, multiple: float, category: str | None
) -> dict[str, Any]:
    scope_name = "overall" if category is None else category
    candidates, excluded = _candidates(models, category)
    ref_block = models[reference_id]
    ref_scope = ref_block if category is None else ref_block["by_category"].get(category)
    ref_cost = None if ref_scope is None else ref_scope["means"]["cost_usd"]
    if ref_cost is None:
        return _insufficient(f"the reference's {scope_name} cost per question is unknown", excluded)
    cap = multiple * ref_cost
    affordable = [c for c in candidates if c["cost"] <= cap]
    if not affordable:
        return _insufficient("no candidate fits the cost cap", excluded)
    best = max(
        affordable,
        # An unknown error count (reference without a clean judged pairing) only matters as a tie-break.
        key=lambda c: (c["win_score"], -(c["unique_factual_errors"] or 0), -c["cost"], c["model_id"]),
    )
    return {
        "model_id": best["model_id"],
        "rationale": (
            f"Highest {scope_name} win score ({best['win_score']:.2f}) among models within "
            f"{multiple:g}x the reference's ${ref_cost:.4f}/question; costs ${best['cost']:.4f}/question"
            f"{_note(excluded)}."
        ),
    }


def recommendation_rules(chat_cost_multiple: float) -> dict[str, str]:
    return {
        "free_trial": (
            f"Cheapest model whose win+tie rate vs the reference is at least {MIN_WIN_TIE_RATE:.0%} "
            "with no unique factual errors (the reference too must show none)."
        ),
        "chat_default": (
            "Highest win score ((wins + 0.5 * ties) / questions) among models costing at most "
            f"{chat_cost_multiple:g}x the reference's mean cost per question."
        ),
        "filing_analysis": (
            f"The chat_default rule restricted to the {FILING_CATEGORY} category "
            "(win score and cost both taken from that category)."
        ),
        "common": (
            "The reference competes as a tie with itself on win score (0.5) and win+tie rate (100%), but its "
            "unique factual errors are read from the judgements: the maximum over its pairings (the same "
            "answers are judged once per pairing), unknown when no pairing was judged cleanly. A question "
            "the model failed to answer while the reference did counts as a loss. Models with judge errors "
            "in scope, no judgement or an unknown cost are excluded, as are models whose run answered no "
            "question; with nothing left the answer is 'insufficient data', never a guess."
        ),
    }


def recommend(
    models: dict[str, dict[str, Any]], reference_id: str, chat_cost_multiple: float
) -> dict[str, Any]:
    rules = recommendation_rules(chat_cost_multiple)
    picks = {
        "chat_default": _pick_best_within_budget(models, reference_id, chat_cost_multiple, None),
        "free_trial": _pick_free_trial(models),
        "filing_analysis": _pick_best_within_budget(models, reference_id, chat_cost_multiple, FILING_CATEGORY),
    }
    return {key: {**pick, "rule": rules[key]} for key, pick in picks.items()}


def build_comparison_summary(
    reference: LoadedRun,
    others: list[LoadedRun],
    judgements: dict[str, Judgement],
    *,
    graph: str,
    judge_model: str,
    run_dirs: dict[str, str],
    chat_cost_multiple: float = DEFAULT_CHAT_COST_MULTIPLE,
    excluded: dict[str, str] | None = None,
) -> dict[str, Any]:
    """``judgements`` maps a non-reference model id to its judgement (A = model, B = reference).

    ``excluded`` maps the id of a model whose run could not be loaded to the reason. Runs in
    ``others`` that answered no question are excluded here too; both end up in
    ``excluded_models`` and as warnings. A reference that answered nothing is an error.
    """
    if (reason := unusable_reason(reference)) is not None:
        raise BenchmarkError(f"The reference run {reference.meta.model_id} cannot anchor a comparison: {reason}")
    excluded_reasons = dict(excluded or {})
    usable: list[LoadedRun] = []
    for run in others:
        if (reason := unusable_reason(run)) is not None:
            excluded_reasons[run.meta.model_id] = reason
        else:
            usable.append(run)
    judgements = {m: j for m, j in judgements.items() if m not in excluded_reasons}

    runs = [reference, *usable]
    base = build_summary(runs)
    blocks = {b["run_id"]: b for b in base["runs"]}
    models = {
        run.meta.model_id: _model_block(
            run, reference, blocks[run.meta.run_id], judgements.get(run.meta.model_id),
            run_dirs[run.meta.model_id],
        )
        for run in runs
    }
    _attach_reference_standing(models, reference.meta.model_id)
    excluded_models = [
        {"model_id": m, "run_dir": run_dirs.get(m), "reason": why} for m, why in excluded_reasons.items()
    ]
    prices_as_of = set(base["prices_as_of"])
    prices_as_of.update(j.judge_prices_as_of for j in judgements.values() if j.judge_prices_as_of)
    runs_usd = sum_known([m["totals"]["cost_usd"] for m in models.values()])
    judge_usd = sum_known([j.judge_cost_usd for j in judgements.values()])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph": graph,
        "reference": reference.meta.model_id,
        "judge_model": judge_model,
        "chat_cost_multiple": chat_cost_multiple,
        "question_set_hash": reference.meta.question_set_hash,
        "question_ids": list(reference.meta.question_ids),
        "prices_as_of": sorted(prices_as_of),
        "warnings": [
            *base["warnings"],
            *(f"{e['model_id']} excluded from the comparison: {e['reason']}" for e in excluded_models),
        ],
        "excluded_models": excluded_models,
        # Everything the comparison cost: every model's run plus all judging. None = unknown
        # (a price or token usage is missing somewhere), never 0.
        "costs": {
            "runs_usd": runs_usd,
            "judge_usd": judge_usd,
            "total_usd": sum_known([runs_usd, judge_usd]),
        },
        "models": models,
        "recommendation_rules": recommendation_rules(chat_cost_multiple),
        "recommendations": recommend(models, reference.meta.model_id, chat_cost_multiple),
    }


_cost = cost_text


def _reference_record(standing: dict[str, Any]) -> str:
    n = standing["pairings_used"]
    if standing["unique_factual_errors"] is None:
        return "reference (unique errors unknown: no cleanly judged pairing)"
    return f"reference (unique errors {standing['unique_factual_errors']}, worst of {n} pairing{'s' if n != 1 else ''})"


def comparison_text(summary: dict[str, Any]) -> str:
    """Terminal digest: per-model line, recommendations and the rules behind them."""
    lines = [f"Reference: {summary['reference']} (judge: {summary['judge_model']})"]
    for model_id, m in summary["models"].items():
        vs = m["vs_reference"]
        if m["is_reference"]:
            record = _reference_record(m["reference_standing"])
        elif vs is None:
            record = "not judged"
        else:
            record = f"W/T/L {vs['wins']}/{vs['ties']}/{vs['losses']}, unique errors {vs['unique_factual_errors']}"
        lines.append(
            f"  {model_id}: {record}; mean {m['means']['llm_calls'] or 0:.1f} LLM calls, "
            f"{m['means']['total_tokens'] or 0:.0f} tokens, {m['means']['latency_s'] or 0:.1f}s, "
            f"{_cost(m['means']['cost_usd'])}/question"
            + ("" if m["judge"] is None else f"; judging {_cost(m['judge']['cost_usd'])}")
        )
    costs = summary["costs"]
    lines.append(
        f"Cost: runs {_cost(costs['runs_usd'])} + judging {_cost(costs['judge_usd'])} "
        f"= total {_cost(costs['total_usd'])}"
    )
    lines.append("Recommendations:")
    for key, rec in summary["recommendations"].items():
        lines.append(f"  {key}: {rec['model_id'] or 'none'} - {rec['rationale']}")
    lines.append("Rules:")
    lines.extend(f"  {key}: {text}" for key, text in summary["recommendation_rules"].items())
    as_of = summary["prices_as_of"]
    lines.append("prices_as_of: " + (", ".join(as_of) if as_of else "unknown (no price data)"))
    lines.extend(f"WARNING: {w}" for w in summary["warnings"])
    return "\n".join(lines)
