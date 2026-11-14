"""``compare-models`` summary: per-model metrics against a reference run, plus recommendations.

Every model's run is judged pairwise against the reference run (A = the model,
B = the reference). The per-model blocks reuse ``summary.build_summary`` for
means, costs and per-category aggregates and add the judge outcome on top.

Recommendation rules (also written verbatim into ``summary.json``):

* The reference competes on a fixed footing: it has no judgement against itself,
  so it counts as a tie on every question (win score 0.5, win+tie rate 100%, no
  unique errors). A challenger can only be recommended over it by beating that.
* ``win score`` = (wins + 0.5 * ties) / questions. A question the model failed to
  answer while the reference answered counts as a loss for the model.
* A model with any judge error in the relevant scope, no judgement, or an unknown
  cost per question is not a candidate; the reason is stated in the rationale.
* ``free_trial``: the cheapest candidate with win+tie rate >= ``MIN_WIN_TIE_RATE``
  and zero unique factual errors.
* ``chat_default``: the candidate with the highest win score among those costing at
  most ``chat_cost_multiple`` x the reference's mean cost per question.
* ``filing_analysis``: the ``chat_default`` rule on the ``filing_lookup`` category only.
* When no candidate remains the answer is ``model_id: null`` with an "insufficient
  data" rationale; nothing is guessed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement
from evals.benchmark.pricing import cost_text
from evals.benchmark.summary import build_summary, judge_usage_block, judgement_counts, sum_known

MIN_WIN_TIE_RATE = 0.85
DEFAULT_CHAT_COST_MULTIPLE = 3.0
FILING_CATEGORY = "filing_lookup"


def _rate(numerator: float, n: int) -> float | None:
    return None if n == 0 else round(numerator / n, 4)


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


def _standing(is_reference: bool, vs: dict[str, Any] | None) -> dict[str, Any] | None:
    """The numbers the rules compare; None when there is no usable judgement."""
    if is_reference:
        return {"win_score": 0.5, "win_or_tie_rate": 1.0, "unique_factual_errors": 0, "judge_errors": 0}
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
        standing = _standing(block["is_reference"], vs)
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


def _pick_free_trial(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates, excluded = _candidates(models, None)
    passing = [
        c for c in candidates
        if c["win_or_tie_rate"] >= MIN_WIN_TIE_RATE and c["unique_factual_errors"] == 0
    ]
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
        key=lambda c: (c["win_score"], -c["unique_factual_errors"], -c["cost"], c["model_id"]),
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
            "with no unique factual errors."
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
            "The reference competes as a tie with itself (win score 0.5). A question the model failed to "
            "answer while the reference did counts as a loss. Models with judge errors in scope, no "
            "judgement or an unknown cost are excluded; with nothing left the answer is "
            "'insufficient data', never a guess."
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
) -> dict[str, Any]:
    """``judgements`` maps a non-reference model id to its judgement (A = model, B = reference)."""
    runs = [reference, *others]
    base = build_summary(runs)
    blocks = {b["run_id"]: b for b in base["runs"]}
    models = {
        run.meta.model_id: _model_block(
            run, reference, blocks[run.meta.run_id], judgements.get(run.meta.model_id),
            run_dirs[run.meta.model_id],
        )
        for run in runs
    }
    prices_as_of = set(base["prices_as_of"])
    prices_as_of.update(j.judge_prices_as_of for j in judgements.values() if j.judge_prices_as_of)
    runs_usd = sum_known([m["totals"]["cost_usd"] for m in models.values()])
    judge_usd = sum_known([j.judge_cost_usd for j in judgements.values()])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph": graph,
        "reference": reference.meta.model_id,
        "judge_model": judge_model,
        "question_set_hash": reference.meta.question_set_hash,
        "question_ids": list(reference.meta.question_ids),
        "prices_as_of": sorted(prices_as_of),
        "warnings": base["warnings"],
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


def comparison_text(summary: dict[str, Any]) -> str:
    """Terminal digest: per-model line, recommendations and the rules behind them."""
    lines = [f"Reference: {summary['reference']} (judge: {summary['judge_model']})"]
    for model_id, m in summary["models"].items():
        vs = m["vs_reference"]
        record = "reference" if vs is None else f"W/T/L {vs['wins']}/{vs['ties']}/{vs['losses']}, unique errors {vs['unique_factual_errors']}"
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
