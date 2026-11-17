"""``summarize``: aggregate runs (and an optional judgement) into one JSON document.

Cost is computed from the price snapshot stored in each ``run.json``. It is
``None`` (unknown) whenever a price is missing or any LLM call had no token
usage; it is never reported as 0.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.benchmark.common import utc_now
from evals.benchmark.pricing import cost_text, estimate_cost, usage_text
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


DELTA_FIELDS = ("llm_calls", "input_tokens", "output_tokens", "total_tokens", "latency_s", "cost_usd")


def _change(old: float | None, new: float | None) -> dict[str, float | None]:
    """``new - old`` and that as a percentage of ``old``; None where either side is unknown.

    The percentage is also None for a zero baseline. An unknown cost stays unknown, never 0.
    """
    if old is None or new is None:
        return {"old": old, "new": new, "delta": None, "pct": None}
    delta = new - old
    return {
        "old": old,
        "new": new,
        "delta": round(delta, 6),
        "pct": None if old == 0 else round(delta / old * 100, 2),
    }


def build_deltas(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Change from the ``old`` run block to the ``new`` one, per question (means) and in total."""
    return {
        "old": old["label"],
        "new": new["label"],
        **{
            scope: {f: _change(old[scope][f], new[scope][f]) for f in DELTA_FIELDS}
            for scope in ("means", "totals")
        },
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


def sum_known(values: list[float | None]) -> float | None:
    """Sum of ``values``, or None when any is unknown (an unknown cost is never counted as 0)."""
    known = [v for v in values if v is not None]
    return round(sum(known), 6) if len(known) == len(values) else None


def judge_usage_block(judgement: Judgement) -> dict[str, Any]:
    """The judge's own spend as recorded in the judgement (None fields on older files)."""
    return {
        "judge_model": judgement.judge_model,
        "input_tokens": judgement.judge_input_tokens,
        "output_tokens": judgement.judge_output_tokens,
        "cost_usd": None if judgement.judge_cost_usd is None else round(judgement.judge_cost_usd, 6),
        "calls_without_usage": judgement.judge_calls_without_usage,
        "prices_as_of": judgement.judge_prices_as_of,
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
        "judge_usage": judge_usage_block(judgement),
        "by_category": dict(by_category),
        # Evidence the judge saw: None date/budget on judgements made before it saw any.
        "as_of_date": judgement.as_of_date,
        "evidence_budget_chars": judgement.evidence_budget_chars,
        "evidence_trimmed_questions": sum(
            bool(q.a_evidence_trimmed or q.b_evidence_trimmed) for q in judgement.questions if q.status == "judged"
        ),
        "evidence_issues": [
            {"id": q.id, "note": q.evidence_issues}
            for q in judgement.questions
            if q.status == "judged" and q.evidence_issues
        ],
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
            # Which questions went wrong: no answer at all, and any with an error recorded along the way.
            "failed_ids": sorted(r.id for r in results if r.status == "failed"),
            "error_ids": sorted(r.id for r in results if r.errors),
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

    prices_as_of = {r.meta.pricing.prices_as_of for r in runs if r.meta.pricing.prices_as_of}
    if judgement is not None and judgement.judge_prices_as_of:
        prices_as_of.add(judgement.judge_prices_as_of)
    runs_usd = sum_known([b["totals"]["cost_usd"] for b in run_blocks])
    judge_usd = None if judgement is None else judgement.judge_cost_usd
    return {
        "generated_at": utc_now(),
        "prices_as_of": sorted(prices_as_of),
        "warnings": warnings,
        "runs": run_blocks,
        # Only for exactly two runs: the first is the old one, the second the new one.
        "deltas": build_deltas(run_blocks[0], run_blocks[1]) if len(run_blocks) == 2 else None,
        "judgement": None if judgement is None else _judgement_block(judgement),
        # Run cost plus the judge's own cost; None = unknown (a price or usage is missing).
        # Without a judgement the total is the runs alone (judge_included is false).
        "costs": {
            "runs_usd": runs_usd,
            "judge_usd": None if judge_usd is None else round(judge_usd, 6),
            "judge_included": judgement is not None,
            "total_usd": runs_usd if judgement is None else sum_known([runs_usd, judge_usd]),
        },
    }




def _signed(change: dict[str, float | None], fmt: str, unit: str = "", prefix: str = "") -> str | None:
    """``+3.0 (+100.0%)``; None when the change is unknown."""
    if change["delta"] is None:
        return None
    sign = "+" if change["delta"] >= 0 else "-"
    pct = "n/a" if change["pct"] is None else f"{change['pct']:+.1f}%"
    return f"{sign}{prefix}{format(abs(change['delta']), fmt)}{unit} ({pct})"


def delta_text(deltas: dict[str, Any]) -> str:
    """One line: the change per question from the old run to the new one."""
    m = deltas["means"]
    parts = []
    for name, field, fmt, unit, prefix in (
        ("LLM calls", "llm_calls", ".1f", "", ""),
        ("input tokens", "input_tokens", ".0f", "", ""),
        ("output tokens", "output_tokens", ".0f", "", ""),
        ("wall time", "latency_s", ".1f", "s", ""),
        ("cost", "cost_usd", ".4f", "", "$"),
    ):
        text = _signed(m[field], fmt, unit, prefix)
        parts.append(f"{name} {text}" if text else f"{name} change unknown")
    return f"{deltas['old']} -> {deltas['new']} (per question): " + ", ".join(parts)


def summary_text(summary: dict[str, Any]) -> str:
    """Short human-readable digest for the terminal (includes prices_as_of)."""
    lines = []
    for run in summary["runs"]:
        m, t = run["means"], run["totals"]
        lines.append(
            f"{run['label']} [{run['architecture']} / {run['model_id']}]: "
            f"{run['questions']['answered']}/{run['questions']['selected']} answered, "
            f"mean {m['llm_calls'] or 0:.1f} LLM calls, {m['total_tokens'] or 0:.0f} tokens, "
            f"{m['latency_s'] or 0:.1f}s; total cost {cost_text(t['cost_usd'])}"
        )
    if summary["deltas"] is not None:
        lines.append(delta_text(summary["deltas"]))
    for run in summary["runs"]:
        lines.append(
            f"{run['label']}: failed {', '.join(run['failed_ids']) or 'none'}; "
            f"with errors {', '.join(run['error_ids']) or 'none'}"
        )
    if summary["judgement"] is not None:
        usage = summary["judgement"]["judge_usage"]
        tokens = usage_text(usage["input_tokens"], usage["output_tokens"], usage["calls_without_usage"])
        lines.append(
            f"judging [{usage['judge_model']}]: cost {cost_text(usage['cost_usd'])} ({tokens}); "
            f"runs + judging total {cost_text(summary['costs']['total_usd'])}"
        )
    as_of = summary["prices_as_of"]
    lines.append("prices_as_of: " + (", ".join(as_of) if as_of else "unknown (no price data)"))
    lines.extend(f"WARNING: {w}" for w in summary["warnings"])
    return "\n".join(lines)
