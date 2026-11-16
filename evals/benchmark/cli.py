"""``uv run python -m evals.benchmark <run|judge|summarize|report|compare-models|compare-report|compare-judgements>``."""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from datetime import datetime
from pathlib import Path

from evals.benchmark.common import DEFAULT_TIMEOUT_S, BenchmarkError, prepare_environment, provider_api_key, resolve_model
from evals.benchmark.compare_summary import DEFAULT_CHAT_COST_MULTIPLE, check_cost_multiple, comparison_text
from evals.benchmark.isolation import quiet_attribute_truncation
from evals.benchmark.questions import DEFAULT_QUESTIONS_PATH, load_questions

# Anchored to the repo root (gitignored), wherever the command is run from.
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "benchmark_reports"


def _ids(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    from evals.benchmark.architectures import ARCHITECTURES

    parser = argparse.ArgumentParser(prog="python -m evals.benchmark", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the question set through one architecture")
    run.add_argument("--graph", required=True, help=f"architecture: {', '.join(sorted(ARCHITECTURES))}")
    run.add_argument("--model", help="registry model id (default: the free-tier model)")
    run.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="question YAML file")
    run.add_argument("--only", type=_ids, help="comma-separated question ids to run")
    run.add_argument("--out", type=Path, required=True, help="output directory (must not hold a run)")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds allowed per question")

    judge = sub.add_parser("judge", help="blind pairwise comparison of two runs")
    judge.add_argument("--a", type=Path, required=True, help="run directory A")
    judge.add_argument("--b", type=Path, required=True, help="run directory B")
    judge.add_argument("--model", help="judge model id (default: the free-tier model)")
    judge.add_argument("--seed", type=int, help="shuffle seed (default: random, recorded in the output)")
    judge.add_argument(
        "--out", type=Path, required=True,
        help="output directory for judgement.json. If it already holds a judgement of the same two runs by "
        "the same judge model, only its errored questions are judged again; any other existing judgement "
        "is refused unless --force",
    )
    judge.add_argument("--force", action="store_true", help="replace a judgement already in --out (re-judges everything)")

    summarize = sub.add_parser("summarize", help="aggregate runs into a JSON summary")
    summarize.add_argument("--runs", type=Path, nargs="+", required=True)
    summarize.add_argument("--judgement", type=Path, help="judgement directory")
    summarize.add_argument("--out", type=Path, required=True, help="summary JSON file")

    report = sub.add_parser("report", help="render a PDF for the human owner")
    report.add_argument("--runs", type=Path, nargs="+", required=True)
    report.add_argument("--judgement", type=Path, help="judgement directory")
    report.add_argument("--out", type=Path, help="PDF file (default: benchmark_reports/benchmark-<timestamp>.pdf in the repo root)")

    compare = sub.add_parser(
        "compare-models",
        help="run the question set per model, judge each against a reference, summarise and write a PDF",
    )
    compare.add_argument("--graph", required=True, help=f"architecture: {', '.join(sorted(ARCHITECTURES))}")
    compare.add_argument(
        "--models", type=_ids, help="comma-separated registry model ids (default: every model whose provider key is set)"
    )
    compare.add_argument("--reference", help="reference model id (default: the free-tier model; run if not in --models)")
    compare.add_argument(
        "--out", type=Path, help="output directory (default: benchmark_reports/compare-<graph>; re-running resumes it)"
    )
    compare.add_argument("--estimate-from", type=Path, help="existing run directory whose token counts drive the cost estimate")
    compare.add_argument(
        "--estimate-judge-from", type=Path,
        help="existing judgement directory whose judge token counts drive the judging cost estimate "
        "(default: judgements already in --out, else an assumption)",
    )
    compare.add_argument("--dry-run", action="store_true", help="print the cost estimate and exit without running anything")
    compare.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    compare.add_argument("--judge-model", help="judge model id (default: the free-tier model)")
    compare.add_argument(
        "--chat-cost-multiple", type=float, default=DEFAULT_CHAT_COST_MULTIPLE,
        help="chat_default / filing_analysis may cost up to this multiple of the reference's cost per question",
    )
    compare.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="question YAML file")
    compare.add_argument("--only", type=_ids, help="comma-separated question ids to run")
    compare.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds allowed per question")

    agreement = sub.add_parser(
        "compare-judgements",
        help="how well two judgements of the same run pair agree (e.g. a cheaper judge vs an expensive one)",
    )
    agreement.add_argument("--a", type=Path, required=True, help="first judgement directory")
    agreement.add_argument("--b", type=Path, required=True, help="second judgement directory (same two runs)")
    agreement.add_argument("--out", type=Path, help="write the full agreement report as JSON to this file")

    compare_report = sub.add_parser(
        "compare-report",
        help="recompute summary.json and report.pdf of a finished compare-models directory from its runs "
        "and judgements on disk (no model is called)",
    )
    compare_report.add_argument("--dir", type=Path, required=True, help="the compare-models --out directory")
    compare_report.add_argument("--out", type=Path, help="PDF file (default: <dir>/report.pdf)")
    compare_report.add_argument("--summary-out", type=Path, help="summary JSON file (default: <dir>/summary.json)")
    compare_report.add_argument(
        "--chat-cost-multiple", type=float,
        help="override the cost multiple recorded in the directory's summary.json",
    )
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from evals.benchmark.runner import preflight, run_benchmark

    prepare_environment()
    model = preflight(args.graph, args.model)
    questions = load_questions(args.questions)
    print(f"Running {args.graph} with model {model.id}")
    with quiet_attribute_truncation():
        asyncio.run(
            run_benchmark(
                args.graph, model, questions, args.out, only=args.only, timeout_s=args.timeout
            )
        )
    return 0


def _previous_judgement(args: argparse.Namespace, run_a, run_b, judge_model_id: str):
    """The judgement already in ``--out`` to complete (re-judging its errors), or None to judge afresh.

    Raises BenchmarkError for an existing judgement that cannot be completed, unless --force.
    """
    from evals.benchmark.judge import errored_ids, rejudge_refusal
    from evals.benchmark.runs import JUDGEMENT_FILE, load_judgement

    path = args.out / JUDGEMENT_FILE
    if args.force or not path.exists():
        return None
    previous = load_judgement(args.out)
    why = rejudge_refusal(previous, run_a, run_b, judge_model_id)
    if why is None and not errored_ids(previous):
        why = "it has no errored question to judge again"
    if why is not None:
        raise BenchmarkError(
            f"{path} already holds a judgement and {why}. Pass --force to replace it (every question is "
            "judged again), or choose another --out."
        )
    return previous


def _cmd_judge(args: argparse.Namespace) -> int:
    from evals.benchmark.judge import build_judge_llm, check_comparable, judge_runs
    from evals.benchmark.pricing import cost_text, pricing_from_model, usage_text
    from evals.benchmark.runs import JUDGEMENT_FILE, load_run, write_model

    prepare_environment()
    model = resolve_model(args.model)
    run_a, run_b = load_run(args.a), load_run(args.b)
    check_comparable(run_a, run_b)
    previous = _previous_judgement(args, run_a, run_b, model.id)  # before any key or LLM is touched
    provider_api_key(model)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    judgement = asyncio.run(
        judge_runs(
            run_a, run_b, build_judge_llm(model), model.id, seed,
            pricing=pricing_from_model(model), previous=previous,
        )
    )
    write_model(args.out / JUDGEMENT_FILE, judgement)
    judged = [q for q in judgement.questions if q.status == "judged"]
    tokens = usage_text(
        judgement.judge_input_tokens, judgement.judge_output_tokens, judgement.judge_calls_without_usage
    )
    print(
        f"Judged {len(judged)}/{len(judgement.questions)} questions "
        f"(A wins {sum(q.winner == 'a' for q in judged)}, B wins {sum(q.winner == 'b' for q in judged)}, "
        f"ties {sum(q.winner == 'tie' for q in judged)}); seed {judgement.seed}. "
        f"Judge usage: {tokens}, cost {cost_text(judgement.judge_cost_usd)}. Wrote {args.out / JUDGEMENT_FILE}"
    )
    return 0


def _load_inputs(args: argparse.Namespace):
    from evals.benchmark.runs import load_judgement, load_run

    runs = [load_run(p) for p in args.runs]
    judgement = load_judgement(args.judgement) if args.judgement else None
    return runs, judgement


def _cmd_summarize(args: argparse.Namespace) -> int:
    from evals.benchmark.runs import write_json
    from evals.benchmark.summary import build_summary, summary_text

    runs, judgement = _load_inputs(args)
    summary = build_summary(runs, judgement)
    write_json(args.out, summary)
    print(summary_text(summary))
    print(f"Wrote {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from evals.benchmark.report import write_report
    from evals.benchmark.summary import build_summary

    runs, judgement = _load_inputs(args)
    out = args.out or DEFAULT_REPORT_DIR / f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.pdf"
    write_report(out, runs, judgement, build_summary(runs, judgement))
    print(f"Wrote {out}")
    return 0


def _cmd_compare_models(args: argparse.Namespace) -> int:
    from evals.benchmark import compare
    from evals.benchmark.judge import build_judge_llm

    check_cost_multiple(args.chat_cost_multiple)
    prepare_environment()
    reference, models = compare.select_models(args.models, args.reference)
    questions = load_questions(args.questions)
    out = args.out or compare.default_out_dir(DEFAULT_REPORT_DIR, args.graph)
    plan = compare.build_plan(args.graph, models, reference, args.judge_model, questions, args.only, out)
    estimate = compare.estimate_plan(plan, args.estimate_from, args.estimate_judge_from)

    print(f"Models: {', '.join(m.id + (' (reference)' if m.id == reference.id else '') for m in models)}")
    print(f"Output directory: {out}")
    print(compare.estimate_text(estimate, len(plan.selected_ids)))
    if args.dry_run:
        print("Dry run: nothing was run.")
        return 0
    if estimate.needs_confirmation and not args.yes:
        if not compare.confirm("Run these models and spend the estimated cost? [y/N] "):
            raise BenchmarkError("Not confirmed, nothing was run. Pass --yes to skip the prompt.")

    with quiet_attribute_truncation():
        summary = asyncio.run(
            compare.execute(
                plan,
                lambda: build_judge_llm(plan.judge_model),
                timeout_s=args.timeout,
                chat_cost_multiple=args.chat_cost_multiple,
            )
        )
    print(comparison_text(summary))
    print(f"Wrote {out / compare.SUMMARY_FILE} and {out / compare.REPORT_FILE}")
    return 0


def _cmd_compare_report(args: argparse.Namespace) -> int:
    from evals.benchmark import compare
    from evals.benchmark.compare_report import write_comparison_report
    from evals.benchmark.runs import write_json

    if args.chat_cost_multiple is not None:
        check_cost_multiple(args.chat_cost_multiple)
    # Recomputed from the runs and judgements on disk (no model call), so the current rules apply.
    summary, runs, judgements = compare.rebuild_comparison(args.dir, args.chat_cost_multiple)
    out = args.out or args.dir / compare.REPORT_FILE
    summary_out = args.summary_out or args.dir / compare.SUMMARY_FILE
    write_json(summary_out, summary)
    write_comparison_report(out, summary, runs, judgements)
    print(comparison_text(summary))
    print(f"Wrote {summary_out} and {out}")
    return 0


def _cmd_compare_judgements(args: argparse.Namespace) -> int:
    from evals.benchmark.agreement import agreement_text, compare_judgements
    from evals.benchmark.runs import load_judgement, write_json

    report = compare_judgements(load_judgement(args.a), load_judgement(args.b))
    print(agreement_text(report))
    if args.out:
        write_json(args.out, report)
        print(f"Wrote {args.out}")
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "judge": _cmd_judge,
    "summarize": _cmd_summarize,
    "report": _cmd_report,
    "compare-models": _cmd_compare_models,
    "compare-report": _cmd_compare_report,
    "compare-judgements": _cmd_compare_judgements,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except BenchmarkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
