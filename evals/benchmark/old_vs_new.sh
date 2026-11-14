#!/usr/bin/env bash
# Benchmark the current graph against the skills agent on the same questions
# and model, judge them blind, and write the summary JSON + owner PDF.
#
# Usage: evals/benchmark/old_vs_new.sh [label] [judge-model]
#   label        suffix for output dirs (default: timestamp)
#   judge-model  registry model id for the judge (default: the CLI's default)
set -euo pipefail

cd "$(dirname "$0")/../.."

label="${1:-$(date +%Y%m%d-%H%M)}"
judge_model="${2:-}"
out="benchmark_reports"
old="$out/runs/current-$label"
new="$out/runs/skills-$label"
judgement="$out/judgements/old-vs-new-$label"

bench() { uv run python -m evals.benchmark "$@"; }

bench run --graph current --out "$old"
bench run --graph skills --out "$new"
bench judge ${judge_model:+--model "$judge_model"} --a "$old" --b "$new" --out "$judgement"
bench summarize --runs "$old" "$new" --judgement "$judgement" --out "$judgement/summary.json"
bench report --runs "$old" "$new" --judgement "$judgement" --out "$out/old-vs-new-$label.pdf"

echo "Summary: $judgement/summary.json"
echo "PDF:     $out/old-vs-new-$label.pdf"
