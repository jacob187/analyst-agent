"""``compare-models``: run the question set per model, judge each against a reference, summarise.

Layout of ``--out``::

    runs/<model-id>/                      one ``run`` per model (resumable: complete runs are skipped)
    judgements/<model-id>-vs-<reference>/ blind pairwise judgement, A = the model, B = the reference
    summary.json                          per-model metrics and recommendations
    report.pdf                            for the human owner

Nothing is ever overwritten: an existing run or judgement is reused only if it
matches this invocation (same architecture, model, question set), otherwise the
command refuses and says what to delete.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agents.model_registry import ModelDef, get_all_models, get_free_tier_model
from evals.benchmark.common import (
    DEFAULT_TIMEOUT_S,
    BenchmarkError,
    has_provider_key,
    provider_api_key,
    resolve_model,
)
from evals.benchmark.compare_report import write_comparison_report
from evals.benchmark.compare_summary import DEFAULT_CHAT_COST_MULTIPLE, build_comparison_summary
from evals.benchmark.judge import judge_runs
from evals.benchmark.pricing import estimate_cost, pricing_from_model
from evals.benchmark.questions import Question, question_set_hash
from evals.benchmark.runner import preflight, run_benchmark
from evals.benchmark.runs import (
    JUDGEMENT_FILE,
    QUESTIONS_DIR,
    RUN_FILE,
    LoadedRun,
    load_judgement,
    load_run,
    write_json,
    write_model,
)
from evals.benchmark.schemas import Judgement

RUNS_DIR = "runs"
JUDGEMENTS_DIR = "judgements"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.pdf"

# Used only when there is no earlier run to take token counts from. A rough guess:
# a multi-step question makes several tool-calling LLM rounds over SEC filing text.
ASSUMED_INPUT_TOKENS_PER_QUESTION = 60_000
ASSUMED_OUTPUT_TOKENS_PER_QUESTION = 4_000
# One judge call: the question, two answers and the instructions.
ASSUMED_JUDGE_INPUT_TOKENS_PER_PAIR = 6_000
ASSUMED_JUDGE_OUTPUT_TOKENS_PER_PAIR = 300


def make_judge_llm(model: ModelDef) -> Any:
    """The judge chat model (the single place the compare command builds a real LLM)."""
    from agents.llm_factory import create_llm

    return create_llm(model.id, provider_api_key(model))


def judgement_dirname(model_id: str, reference_id: str) -> str:
    return f"{model_id}-vs-{reference_id}"


def select_models(
    model_ids: list[str] | None, reference_id: str | None
) -> tuple[ModelDef, list[ModelDef]]:
    """(reference, models) with the reference first and never duplicated.

    ``model_ids`` defaults to every registry model whose provider key is in the
    environment; the reference defaults to the free-tier model and is added to
    the list when missing from it.
    """
    reference = resolve_model(reference_id)
    if model_ids:
        chosen = [resolve_model(i) for i in dict.fromkeys(model_ids)]
    else:
        chosen = [m for m in get_all_models() if has_provider_key(m)]
        if not chosen:
            raise BenchmarkError(
                "No registry model has its provider API key in the environment; "
                "set a key or pass --models explicitly."
            )
    return reference, [reference, *[m for m in chosen if m.id != reference.id]]


@dataclass
class RunPlan:
    model: ModelDef
    dir: Path
    complete: bool


def plan_run(model: ModelDef, out: Path, graph: str, questions: list[Question], selected_ids: list[str]) -> RunPlan:
    """Decide whether ``model``'s run is still to do, or complete and reusable.

    Raises BenchmarkError for a directory that holds anything else: an
    incomplete run (cannot be resumed mid-way) or a complete run of different
    work (other architecture, model or question set).
    """
    run_dir = out / RUNS_DIR / model.id
    if not (run_dir / RUN_FILE).exists():
        if run_dir.exists() and any(run_dir.iterdir()):
            raise BenchmarkError(f"{run_dir} exists but holds no {RUN_FILE}; remove it or choose a new --out.")
        return RunPlan(model, run_dir, complete=False)

    run = load_run(run_dir)
    done = all((run_dir / QUESTIONS_DIR / f"{qid}.json").exists() for qid in run.meta.question_ids)
    if run.meta.finished_at is None or not done:
        raise BenchmarkError(
            f"{run_dir} holds an incomplete run (interrupted?). Runs cannot resume mid-way: "
            f"delete {run_dir} and re-run to redo this model."
        )
    expected = {
        "architecture": graph,
        "model_id": model.id,
        "question_set_hash": question_set_hash(questions),
        "question_ids": selected_ids,
    }
    stale = [k for k, v in expected.items() if getattr(run.meta, k) != v]
    if stale:
        raise BenchmarkError(
            f"{run_dir} holds a complete run that differs from this invocation ({', '.join(stale)}). "
            f"Delete {run_dir} to redo it, or choose a new --out."
        )
    return RunPlan(model, run_dir, complete=True)


@dataclass
class Baseline:
    input_tokens: float
    output_tokens: float
    source: str
    rough: bool


def baseline_tokens(estimate_from: Path | None, reference_plan: RunPlan) -> Baseline:
    """Mean tokens per question: ``--estimate-from``, else the reference run, else an assumption."""
    source_dir = estimate_from
    label = f"run {estimate_from}"
    if source_dir is None and reference_plan.complete:
        source_dir, label = reference_plan.dir, f"the reference run {reference_plan.dir}"
    if source_dir is not None:
        run = load_run(source_dir)
        results = list(run.questions.values())
        if results:
            note = ""
            if any(r.llm_calls_without_usage for r in results):
                note = " (some calls reported no usage: a lower bound)"
            return Baseline(
                sum(r.input_tokens for r in results) / len(results),
                sum(r.output_tokens for r in results) / len(results),
                f"mean tokens per question from {label}{note}",
                rough=False,
            )
        if estimate_from is not None:
            raise BenchmarkError(f"--estimate-from {estimate_from} has no recorded questions.")
    return Baseline(
        ASSUMED_INPUT_TOKENS_PER_QUESTION,
        ASSUMED_OUTPUT_TOKENS_PER_QUESTION,
        "ROUGH GUESS: no earlier run to measure; assuming "
        f"{ASSUMED_INPUT_TOKENS_PER_QUESTION:,} input / {ASSUMED_OUTPUT_TOKENS_PER_QUESTION:,} output tokens per question",
        rough=True,
    )


@dataclass
class CostLine:
    label: str
    model_id: str
    usd: float | None
    pending: bool
    price_as_of: str | None = None


@dataclass
class Estimate:
    baseline: Baseline
    lines: list[CostLine]
    free_tier_id: str | None

    @property
    def total_known(self) -> float:
        return sum(line.usd for line in self.lines if line.pending and line.usd is not None)

    @property
    def unknown(self) -> int:
        return sum(line.pending and line.usd is None for line in self.lines)

    @property
    def needs_confirmation(self) -> bool:
        """Anything pending beyond the free-tier model must be confirmed (also when its price is unknown)."""
        return any(line.pending and line.model_id != self.free_tier_id for line in self.lines)

    @property
    def has_pending(self) -> bool:
        return any(line.pending for line in self.lines)


def build_estimate(
    run_plans: list[RunPlan],
    judges_pending: int,
    judge_model: ModelDef,
    n_questions: int,
    baseline: Baseline,
) -> Estimate:
    lines: list[CostLine] = []
    for plan in run_plans:
        pricing = pricing_from_model(plan.model)
        usd = estimate_cost(
            pricing, round(baseline.input_tokens * n_questions), round(baseline.output_tokens * n_questions)
        )
        lines.append(CostLine(f"run {plan.model.id}", plan.model.id, usd, not plan.complete, pricing.prices_as_of))
    if judges_pending:
        pricing = pricing_from_model(judge_model)
        usd = estimate_cost(
            pricing,
            ASSUMED_JUDGE_INPUT_TOKENS_PER_PAIR * n_questions * judges_pending,
            ASSUMED_JUDGE_OUTPUT_TOKENS_PER_PAIR * n_questions * judges_pending,
        )
        lines.append(
            CostLine(
                f"judge x{judges_pending} (assumed {ASSUMED_JUDGE_INPUT_TOKENS_PER_PAIR:,} in / "
                f"{ASSUMED_JUDGE_OUTPUT_TOKENS_PER_PAIR:,} out tokens per question)",
                judge_model.id, usd, True, pricing.prices_as_of,
            )
        )
    free = get_free_tier_model()
    return Estimate(baseline, lines, free.id if free else None)


def estimate_text(estimate: Estimate, n_questions: int) -> str:
    out = [f"Estimated cost ({n_questions} questions per model; {estimate.baseline.source}):"]
    for line in estimate.lines:
        if not line.pending:
            out.append(f"  {line.label}: already complete, skipped")
            continue
        cost = "unknown (price missing)" if line.usd is None else f"${line.usd:.4f}"
        as_of = f" [prices as of {line.price_as_of}]" if line.price_as_of else ""
        out.append(f"  {line.label}: {cost}{as_of}")
    total = f"${estimate.total_known:.4f}"
    if estimate.unknown:
        total += f" for the priced items; {estimate.unknown} item(s) have unknown cost, so the real total is unknown"
    out.append(f"  TOTAL: {total}")
    if estimate.baseline.rough:
        out.append("  (the token counts are a rough guess; the real cost may differ a lot)")
    return "\n".join(out)


def confirm(prompt: str) -> bool:
    """True only for an explicit yes. EOF or an unreadable stdin counts as no."""
    try:
        answer = input(prompt)
    except (EOFError, OSError):
        return False
    return answer.strip().lower() in {"y", "yes"}


def default_out_dir(report_dir: Path, graph: str) -> Path:
    """Stable per graph, so re-running the same command finds and resumes its runs."""
    return report_dir / f"compare-{graph}"


@dataclass
class Plan:
    graph: str
    reference: ModelDef
    run_plans: list[RunPlan]
    judge_model: ModelDef
    out: Path
    questions: list[Question]
    selected_ids: list[str]
    only: list[str] | None
    judges_pending: list[ModelDef] = field(default_factory=list)


def build_plan(
    graph: str,
    models: list[ModelDef],
    reference: ModelDef,
    judge_model_id: str | None,
    questions: list[Question],
    only: list[str] | None,
    out: Path,
) -> Plan:
    """Validate everything that can be validated without spending: flags, keys, existing output."""
    for model in models:
        preflight(graph, model.id)  # unknown architecture, or the model's provider key is missing
    judge_model = resolve_model(judge_model_id)

    selected = questions
    if only:
        unknown = [i for i in only if i not in {q.id for q in questions}]
        if unknown:
            raise BenchmarkError(f"Unknown question id(s) in --only: {', '.join(unknown)}")
        selected = [q for q in questions if q.id in set(only)]
    selected_ids = [q.id for q in selected]

    run_plans = [plan_run(m, out, graph, questions, selected_ids) for m in models]
    by_id = {p.model.id: p for p in run_plans}
    for model in models:
        path = _judgement_file(out, model, reference)
        if model.id == reference.id or not path.exists():
            continue
        # An existing judgement is reusable only if it judged exactly the runs that will be summarised.
        if not (by_id[model.id].complete and by_id[reference.id].complete):
            raise BenchmarkError(
                f"{path.parent} holds a judgement but a run it needs is missing. "
                f"Delete {path.parent} (or restore the runs) and re-run."
            )
        _check_judgement(load_judgement(path.parent), load_run(by_id[model.id].dir), load_run(by_id[reference.id].dir), path)
    pending = [m for m in models if m.id != reference.id and not _judgement_file(out, m, reference).exists()]
    if pending:
        provider_api_key(judge_model)
    return Plan(graph, reference, run_plans, judge_model, out, questions, selected_ids, only, pending)


def _judgement_file(out: Path, model: ModelDef, reference: ModelDef) -> Path:
    return out / JUDGEMENTS_DIR / judgement_dirname(model.id, reference.id) / JUDGEMENT_FILE


def estimate_plan(plan: Plan, estimate_from: Path | None) -> Estimate:
    reference_plan = next(p for p in plan.run_plans if p.model.id == plan.reference.id)
    baseline = baseline_tokens(estimate_from, reference_plan)
    return build_estimate(
        plan.run_plans, len(plan.judges_pending), plan.judge_model, len(plan.selected_ids), baseline
    )


def _check_judgement(judgement: Judgement, model_run: LoadedRun, reference_run: LoadedRun, path: Path) -> None:
    if judgement.a.run_id != model_run.meta.run_id or judgement.b.run_id != reference_run.meta.run_id:
        raise BenchmarkError(
            f"{path} judges different runs than the ones in {model_run.dir.parent}. "
            f"Delete {path.parent} to re-judge."
        )


async def execute(
    plan: Plan,
    judge_llm_factory: Callable[[], Any],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    chat_cost_multiple: float = DEFAULT_CHAT_COST_MULTIPLE,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run what is missing, judge what is missing, write ``summary.json`` and ``report.pdf``."""
    for run_plan in plan.run_plans:
        if run_plan.complete:
            progress(f"{run_plan.model.id}: complete run found at {run_plan.dir}; skipping")
            continue
        progress(f"Running {plan.graph} with model {run_plan.model.id}")
        await run_benchmark(
            plan.graph, run_plan.model, plan.questions, run_plan.dir,
            only=plan.only, timeout_s=timeout_s, progress=progress,
        )

    runs = {p.model.id: load_run(p.dir) for p in plan.run_plans}
    reference_run = runs[plan.reference.id]
    judgements: dict[str, Judgement] = {}
    judge_llm = None
    for model_id, model_run in runs.items():
        if model_id == plan.reference.id:
            continue
        model = next(p.model for p in plan.run_plans if p.model.id == model_id)
        path = _judgement_file(plan.out, model, plan.reference)
        if path.exists():
            judgement = load_judgement(path.parent)  # already checked against these runs by build_plan
            progress(f"{model_id}: judgement found at {path.parent}; skipping")
        else:
            judge_llm = judge_llm or judge_llm_factory()
            progress(f"Judging {model_id} against {plan.reference.id}")
            judgement = await judge_runs(
                model_run, reference_run, judge_llm, plan.judge_model.id, secrets.randbits(32), progress=progress
            )
            write_model(path, judgement)
            errors = sum(q.status == "error" for q in judgement.questions)
            if errors:
                progress(f"{model_id}: {errors} judge error(s); delete {path.parent} to re-judge")
        judgements[model_id] = judgement

    others = [r for model_id, r in runs.items() if model_id != plan.reference.id]
    summary = build_comparison_summary(
        reference_run, others, judgements,
        graph=plan.graph,
        judge_model=plan.judge_model.id,
        run_dirs={model_id: str(Path(RUNS_DIR) / model_id) for model_id in runs},
        chat_cost_multiple=chat_cost_multiple,
    )
    write_json(plan.out / SUMMARY_FILE, summary)
    write_comparison_report(plan.out / REPORT_FILE, summary, runs, judgements)
    return summary


def load_comparison(out: Path) -> tuple[dict[str, Any], dict[str, LoadedRun], dict[str, Judgement]]:
    """Reload a finished comparison directory (for ``report --comparison``)."""
    summary_path = out / SUMMARY_FILE
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BenchmarkError(f"Cannot read {summary_path}: {e}") from e
    reference = summary["reference"]
    runs = {model_id: load_run(out / block["run_dir"]) for model_id, block in summary["models"].items()}
    judgements = {
        model_id: load_judgement(out / JUDGEMENTS_DIR / judgement_dirname(model_id, reference))
        for model_id in summary["models"]
        if model_id != reference
    }
    return summary, runs, judgements
