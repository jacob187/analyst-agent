"""``uv run python -m evals.benchmark <run|judge|summarize|report>``."""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from datetime import datetime
from pathlib import Path

from evals.benchmark.common import DEFAULT_TIMEOUT_S, BenchmarkError, prepare_environment, provider_api_key, resolve_model
from evals.benchmark.questions import DEFAULT_QUESTIONS_PATH, load_questions

DEFAULT_REPORT_DIR = Path("benchmark_reports")


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
    judge.add_argument("--out", type=Path, required=True, help="output directory for judgement.json")

    summarize = sub.add_parser("summarize", help="aggregate runs into a JSON summary")
    summarize.add_argument("--runs", type=Path, nargs="+", required=True)
    summarize.add_argument("--judgement", type=Path, help="judgement directory")
    summarize.add_argument("--out", type=Path, required=True, help="summary JSON file")

    report = sub.add_parser("report", help="render a PDF for the human owner")
    report.add_argument("--runs", type=Path, nargs="+", required=True)
    report.add_argument("--judgement", type=Path, help="judgement directory")
    report.add_argument("--out", type=Path, help=f"PDF file (default: {DEFAULT_REPORT_DIR}/benchmark-<timestamp>.pdf)")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from evals.benchmark.runner import preflight, run_benchmark

    prepare_environment()
    _, model = preflight(args.graph, args.model)
    questions = load_questions(args.questions)
    print(f"Running {args.graph} with model {model.id}")
    asyncio.run(
        run_benchmark(
            args.graph, model, questions, args.out, only=args.only, timeout_s=args.timeout
        )
    )
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from agents.llm_factory import create_llm
    from evals.benchmark.judge import check_comparable, judge_runs
    from evals.benchmark.runs import JUDGEMENT_FILE, load_run, write_model

    prepare_environment()
    model = resolve_model(args.model)
    api_key = provider_api_key(model)
    run_a, run_b = load_run(args.a), load_run(args.b)
    check_comparable(run_a, run_b)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    judgement = asyncio.run(judge_runs(run_a, run_b, create_llm(model.id, api_key), model.id, seed))
    write_model(args.out / JUDGEMENT_FILE, judgement)
    judged = [q for q in judgement.questions if q.status == "judged"]
    print(
        f"Judged {len(judged)}/{len(judgement.questions)} questions "
        f"(A wins {sum(q.winner == 'a' for q in judged)}, B wins {sum(q.winner == 'b' for q in judged)}, "
        f"ties {sum(q.winner == 'tie' for q in judged)}); seed {seed}. Wrote {args.out / JUDGEMENT_FILE}"
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


_COMMANDS = {
    "run": _cmd_run,
    "judge": _cmd_judge,
    "summarize": _cmd_summarize,
    "report": _cmd_report,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except BenchmarkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
