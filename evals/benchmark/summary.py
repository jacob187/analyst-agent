"""``summarize``: aggregate runs (and an optional judgement) into one JSON document.

Cost is computed from the price snapshot stored in each ``run.json``. It is
``None`` (unknown) whenever a price is missing or any LLM call had no token
usage; it is never reported as 0.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from evals.benchmark.pricing import estimate_cost
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement, Pricing, QuestionResult


def _aggregate(results: list[QuestionResult], pricing: Pricing) -> dict[str, Any]:
    n = len(results)
    costs = [
        estimate_cost(pricing, r.input_tokens, r.output_tokens, r.llm_calls_without_usage)
        for r in results
    ]
    unknown = sum(c is None for c in costs)
    known_sum = sum(c for c in costs if c is not None)
    total_cost = known_sum if n and unknown == 0 else None

    def total(field: str) -> float:
        return sum(getattr(r, field) for r in results)

    def mean(value: float | None) -> float | None:
        return None if value is None or n == 0 else value / n

    totals = {
        "llm_calls": int(total("llm_calls")),
        "input_tokens": int(total("input_tokens")),
        "output_tokens": int(total("output_tokens")),
        "total_tokens": int(total("total_tokens")),
        "latency_s": round(total("latency_s"), 3),
        "cost_usd": None if total_cost is None else round(total_cost, 6),
    }
    means = {k: (None if (m := mean(v)) is None else round(m, 6)) for k, v in totals.items()}
    return {
        "n": n,
        "totals": totals,
        "means": means,
        "cost_unknown_questions": unknown,
        "cost_known_usd": round(known_sum, 6),
        "llm_calls_without_usage": int(total("llm_calls_without_usage")),
    }


def judgement_counts(judgement: Judgement, side: str) -> dict[str, int]:
    other = "b" if side == "a" else "a"
    judged = [q for q in judgement.questions if q.status == "judged"]
    return {
        "judged": len(judged),
        "skipped_or_errored": len(judgement.questions) - len(judged),
        "wins": sum(q.winner == side for q in judged),
        "losses": sum(q.winner == other for q in judged),
        "ties": sum(q.winner == "tie" for q in judged),
        "unique_factual_errors": sum(bool(getattr(q, f"{side}_unique_factual_error")) for q in judged),
        "opponent_unique_factual_errors": sum(
            bool(getattr(q, f"{other}_unique_factual_error")) for q in judged
        ),
    }


def _judgement_block(judgement: Judgement) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0})
    for q in judgement.questions:
        if q.status != "judged":
            continue
        key = {"a": "a_wins", "b": "b_wins", "tie": "ties"}[q.winner]
        by_category[q.category][key] += 1
    return {
        "judge_model": judgement.judge_model,
        "seed": judgement.seed,
        "date": judgement.date,
        "question_set_hash": judgement.question_set_hash,
        "a": judgement.a.model_dump(),
        "b": judgement.b.model_dump(),
        "a_counts": judgement_counts(judgement, "a"),
        "b_counts": judgement_counts(judgement, "b"),
        "by_category": dict(by_category),
    }


def build_summary(runs: list[LoadedRun], judgement: Judgement | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    if len({r.meta.question_set_hash for r in runs}) > 1:
        warnings.append("Runs used different question sets; their numbers are not directly comparable.")
    if len({r.meta.tavily_enabled for r in runs}) > 1:
        warnings.append("Runs differ in whether web research (Tavily) was enabled.")
    if any(r.meta.git_dirty for r in runs):
        warnings.append("At least one run was made from a working tree with uncommitted changes.")

    run_blocks: list[dict[str, Any]] = []
    for run in runs:
        meta = run.meta
        results = [run.questions[qid] for qid in meta.question_ids if qid in run.questions]
        block: dict[str, Any] = {
            "label": run.label,
            "dir": str(run.dir),
            "run_id": meta.run_id,
            "architecture": meta.architecture,
            "model_id": meta.model_id,
            "git_sha": meta.git_sha,
            "git_dirty": meta.git_dirty,
            "date": meta.date,
            "question_set_hash": meta.question_set_hash,
            "tavily_enabled": meta.tavily_enabled,
            "pricing": meta.pricing.model_dump(),
            "questions": {
                "selected": len(meta.question_ids),
                "recorded": len(results),
                "answered": sum(r.status == "ok" for r in results),
                "failed": sum(r.status == "failed" for r in results),
                "with_errors": sum(bool(r.errors) for r in results),
            },
            **_aggregate(results, meta.pricing),
        }
        by_cat: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in results:
            by_cat[r.category].append(r)
        block["by_category"] = {
            cat: _aggregate(rs, meta.pricing) for cat, rs in sorted(by_cat.items())
        }
        block["judgement"] = None
        if judgement is not None and meta.run_id in (judgement.a.run_id, judgement.b.run_id):
            block["judgement"] = judgement_counts(judgement, "a" if meta.run_id == judgement.a.run_id else "b")
        run_blocks.append(block)

    if judgement is not None and not any(b["judgement"] for b in run_blocks):
        warnings.append("The judgement does not match any of the given runs (run_id mismatch).")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prices_as_of": sorted({r.meta.pricing.prices_as_of for r in runs if r.meta.pricing.prices_as_of}),
        "warnings": warnings,
        "runs": run_blocks,
        "judgement": None if judgement is None else _judgement_block(judgement),
    }


def _cost_text(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def summary_text(summary: dict[str, Any]) -> str:
    """Short human-readable digest for the terminal (includes prices_as_of)."""
    lines = []
    for run in summary["runs"]:
        m, t = run["means"], run["totals"]
        lines.append(
            f"{run['label']} [{run['architecture']} / {run['model_id']}]: "
            f"{run['questions']['answered']}/{run['questions']['selected']} answered, "
            f"mean {m['llm_calls'] or 0:.1f} LLM calls, {m['total_tokens'] or 0:.0f} tokens, "
            f"{m['latency_s'] or 0:.1f}s; total cost {_cost_text(t['cost_usd'])}"
        )
    as_of = summary["prices_as_of"]
    lines.append("prices_as_of: " + (", ".join(as_of) if as_of else "unknown (no price data)"))
    lines.extend(f"WARNING: {w}" for w in summary["warnings"])
    return "\n".join(lines)
