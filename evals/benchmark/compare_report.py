"""PDF for ``compare-models`` (the human owner's view; JSON stays the machine interface).

Order: per-model table, recommendations with their rules, per-category table, answers
side by side for one representative question per category, then an appendix listing
every question with each model's outcome and the remaining questions' answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.benchmark.compare_summary import judge_label
from evals.benchmark.pricing import cost_text
from evals.benchmark.report import (
    _columns,
    _heading,
    _metrics_line,
    _num,
    _paragraph,
    _Pdf,
    _table,
)
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement, JudgedQuestion, QuestionResult

# Answer columns per row-group; more models than this continue in the next group.
MODELS_PER_ROW = 3


def representative_ids(summary: dict[str, Any], results: dict[str, QuestionResult]) -> list[str]:
    """The first question (in question-set order) of each category."""
    seen: dict[str, str] = {}
    for qid in summary["question_ids"]:
        result = results.get(qid)
        if result is not None:
            seen.setdefault(result.category, qid)
    return list(seen.values())


def _outcome(entry: JudgedQuestion | None) -> str:
    if entry is None:
        return "-"
    if entry.status == "error":
        return "err"
    if entry.status == "skipped":
        return "no ans"
    word = {"a": "W", "tie": "T", "b": "L"}[entry.winner]
    return word + ("*" if entry.a_unique_factual_error else "")


def _entry(judgement: Judgement | None, qid: str) -> JudgedQuestion | None:
    if judgement is None:
        return None
    return next((q for q in judgement.questions if q.id == qid), None)


def _result_for(runs: dict[str, LoadedRun], qid: str) -> QuestionResult | None:
    return next((r.questions[qid] for r in runs.values() if qid in r.questions), None)


def _question_section(
    pdf: _Pdf,
    qid: str,
    summary: dict[str, Any],
    runs: dict[str, LoadedRun],
    judgements: dict[str, Judgement],
) -> None:
    reference = summary["reference"]
    pdf.add_page()
    _heading(pdf, qid, 14)
    ref_result = _result_for(runs, qid)
    if ref_result is not None:
        _paragraph(pdf, f"{ref_result.ticker} | {ref_result.category}", style="I")
        _paragraph(pdf, f"Question: {ref_result.question}", size=10, style="B")
    pdf.ln(1)
    for model_id, run in runs.items():
        _paragraph(pdf, _metrics_line(run, run.questions.get(qid)), size=8)
    pdf.ln(1)
    for model_id in runs:
        if model_id == reference:
            continue
        entry = _entry(judgements.get(model_id), qid)
        if entry is None or entry.status != "judged":
            line = f"{model_id} vs {reference}: {_outcome(entry)} ({entry.reason if entry and entry.reason else 'not judged'})"
        else:
            verdict = {"a": f"{model_id} wins", "b": f"{reference} wins", "tie": "tie"}[entry.winner]
            line = (
                f"{model_id} vs {reference}: {verdict}; unique factual error - "
                f"{model_id}: {'YES' if entry.a_unique_factual_error else 'no'}, "
                f"{reference}: {'YES' if entry.b_unique_factual_error else 'no'}. {entry.rationale or ''}"
            )
        _paragraph(pdf, line, size=8)
    pdf.ln(2)
    ids = list(runs)
    for start in range(0, len(ids), MODELS_PER_ROW):
        group = ids[start:start + MODELS_PER_ROW]
        _columns(
            pdf,
            [model_id + (" (reference)" if model_id == reference else "") for model_id in group],
            [
                (runs[m].questions[qid].answer if qid in runs[m].questions and runs[m].questions[qid].answer else "(no answer)")
                for m in group
            ],
        )
        pdf.ln(2)


def render_comparison_report(
    summary: dict[str, Any], runs: dict[str, LoadedRun], judgements: dict[str, Judgement]
) -> bytes:
    pdf = _Pdf(orientation="L", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_title("Analyst Agent model comparison")
    pdf.add_page()

    reference = summary["reference"]
    models = summary["models"]
    runs = {m: r for m, r in runs.items() if m in models}  # excluded models have no answers worth showing
    ids = list(models)
    _heading(pdf, "Analyst Agent model comparison", 18)
    as_of = summary["prices_as_of"]
    judges = judge_label(summary)
    _paragraph(
        pdf,
        f"Generated {summary['generated_at']}. Architecture: {summary['graph']}. "
        f"{judges[0].upper()}{judges[1:]}. "
        "Prices as of: " + (", ".join(as_of) if as_of else "unknown (no price data; costs shown as unknown)"),
    )
    _paragraph(pdf, "Models: " + "; ".join(m + (" (reference)" if m == reference else "") for m in ids))
    _paragraph(
        pdf,
        "Each model's answers were judged blind and pairwise against the reference. W/T/L is from the "
        "model's side; a question the model failed to answer counts as a loss. Outcome '*' = the judge "
        "found a factual error only in that model's answer.",
        style="I",
    )
    for warning in summary["warnings"]:
        _paragraph(pdf, f"WARNING: {warning}", style="B")
    pdf.ln(2)

    excluded = summary["excluded_models"]
    if excluded:
        _heading(pdf, "Excluded models (no usable run)", 12)
        for entry in excluded:
            spent = "" if entry["cost_usd"] is None else f" (run cost {cost_text(entry['cost_usd'])}, counted below)"
            _paragraph(pdf, f"{entry['model_id']}: {entry['reason']}{spent}")
        pdf.ln(2)

    _heading(pdf, "Per-model summary (means per question)", 12)
    rows = []
    for model_id, m in models.items():
        vs, means, q = m["vs_reference"], m["means"], m["questions"]
        standing = m.get("reference_standing")  # only the reference's block has one
        ref_errors = (
            "-" if standing is None
            else f"unknown ({standing['answers_judged']}/{standing['answers']} judged)"
            if standing["unique_factual_errors"] is None
            else f"{standing['unique_factual_errors']} (distinct)"
        )
        rows.append([
            model_id + (" (ref)" if m["is_reference"] else ""),
            f"{q['answered']}/{q['selected']}",
            "-" if vs is None else f"{vs['wins']}/{vs['ties']}/{vs['losses'] + vs['unanswered_by_model']}",
            "-" if vs is None or vs["win_or_tie_rate"] is None else f"{vs['win_or_tie_rate']:.0%}",
            ref_errors if vs is None else f"{vs['unique_factual_errors']} / {vs['reference_unique_factual_errors']}",
            "-" if vs is None else str(vs["judge_errors"]),
            _num(means["llm_calls"], ".1f"), _num(means["total_tokens"], ".0f"),
            _num(means["latency_s"], ".1f"), cost_text(means["cost_usd"]),
            "-" if m["judge"] is None else cost_text(m["judge"]["cost_usd"]),
            m["prices_as_of"] or "unknown",
        ])
    _table(
        pdf,
        ["Model", "Answered", "W/T/L", "Win+tie", "Unique errors (own / ref)", "Judge errors",
         "LLM calls", "Tokens", "Latency s", "Cost / question", "Judging cost", "Prices as of"],
        rows,
        [44, 17, 17, 15, 26, 15, 15, 17, 17, 20, 20, 21],
    )
    costs = summary["costs"]
    excluded_usd = costs["excluded_runs_usd"]
    _paragraph(
        pdf,
        f"Run cost: {cost_text(costs['runs_usd'])}"
        + ("" if excluded_usd == 0 else f" (incl. {cost_text(excluded_usd)} for excluded runs)")
        + f". Judge cost: {cost_text(costs['judge_usd'])}. "
        f"Total (runs + judging): {cost_text(costs['total_usd'])}.",
        style="B",
    )
    pdf.ln(3)

    _heading(pdf, "Recommendations", 12)
    for key, rec in summary["recommendations"].items():
        _paragraph(pdf, f"{key}: {rec['model_id'] or 'none'}", style="B")
        _paragraph(pdf, rec["rationale"])
        _paragraph(pdf, f"Rule: {rec['rule']}", size=8, style="I")
    _paragraph(pdf, summary["recommendation_rules"]["common"], size=8, style="I")
    pdf.ln(2)

    _heading(pdf, "By category (means per question)", 12)
    cat_rows = []
    for model_id, m in models.items():
        for cat, agg in m["by_category"].items():
            vs = agg["vs_reference"]
            cat_rows.append([
                model_id, cat, str(agg["n"]),
                "-" if vs is None else f"{vs['wins']}/{vs['ties']}/{vs['losses'] + vs['unanswered_by_model']}",
                _num(agg["means"]["llm_calls"], ".1f"), _num(agg["means"]["total_tokens"], ".0f"),
                _num(agg["means"]["latency_s"], ".1f"), cost_text(agg["means"]["cost_usd"]),
            ])
    _table(
        pdf, ["Model", "Category", "N", "W/T/L", "LLM calls", "Tokens", "Latency s", "Mean cost"],
        cat_rows, [50, 30, 10, 20, 20, 22, 22, 24],
    )

    results = {qid: r for qid in summary["question_ids"] if (r := _result_for(runs, qid)) is not None}
    featured = representative_ids(summary, results)
    pdf.add_page()
    _heading(pdf, "Representative answers (one question per category)", 14)
    _paragraph(pdf, "Questions: " + ", ".join(featured))
    for qid in featured:
        _question_section(pdf, qid, summary, runs, judgements)

    pdf.add_page()
    _heading(pdf, "Appendix A: all questions", 14)
    _paragraph(
        pdf,
        "Columns: " + "; ".join(f"M{i} = {m}" for i, m in enumerate(ids, start=1))
        + ". Cells are the outcome against the reference ('ref' = the reference itself).",
    )
    rows = []
    for qid in summary["question_ids"]:
        r = results.get(qid)
        rows.append([
            qid, "?" if r is None else r.category, "?" if r is None else r.ticker,
            "?" if r is None else r.question,
            *["ref" if m == reference else _outcome(_entry(judgements.get(m), qid)) for m in ids],
        ])
    _table(
        pdf,
        ["Question", "Category", "Ticker", "Text", *[f"M{i}" for i in range(1, len(ids) + 1)]],
        rows,
        [30, 26, 14, 120, *[14] * len(ids)],
    )

    rest = [qid for qid in summary["question_ids"] if qid not in featured]
    if rest:
        pdf.add_page()
        _heading(pdf, "Appendix B: answers to the remaining questions", 14)
        for qid in rest:
            _question_section(pdf, qid, summary, runs, judgements)

    return bytes(pdf.output())


def write_comparison_report(
    path: Path | str, summary: dict[str, Any], runs: dict[str, LoadedRun], judgements: dict[str, Judgement]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_comparison_report(summary, runs, judgements))
    return path
