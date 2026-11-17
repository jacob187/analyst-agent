"""AC2: the judge's own cost shows in the summary JSON, the terminal digests and both PDFs,
and the compare-models total is runs plus judging.

Runs are fixture files with the real schema; judgements come from the real judge driven by
`FakeJudge(usage=...)` (a scripted chat model reporting token usage), so the cost is genuinely
measured from spans. No model or network is touched.
"""

import json

import pytest

from evals.benchmark.cli import main
from evals.benchmark.compare_report import render_comparison_report
from evals.benchmark.compare_summary import build_comparison_summary, comparison_text
from evals.benchmark.judge import judge_runs
from evals.benchmark.report import render_report
from evals.benchmark.runs import load_run, write_model
from evals.benchmark.summary import build_summary, summary_text
from tests.benchmark.helpers import PRICED, UNPRICED, FakeJudge, pdf_text_blob, write_run

pytestmark = pytest.mark.eval_unit

# 5 questions per run; each run costs 5 * (1000 * 2 + 200 * 10) / 1e6 = 0.02 at PRICED.
RUN_COST = 0.02
# 5 judge calls of 3000 in / 100 out at $4 / $20 per Mtok: 5 * (3000 * 4 + 100 * 20) / 1e6.
JUDGE_PRICING = PRICED.model_copy(update={"input_price_per_mtok": 4.0, "output_price_per_mtok": 20.0, "prices_as_of": "2026-03-01"})
JUDGE_COST = 0.07


async def _judged(tmp_path, *, usage=(3000, 100), pricing=JUDGE_PRICING, run_b_pricing=PRICED):
    a = load_run(write_run(tmp_path, "run-a", model_id="model-a"))
    b = load_run(write_run(tmp_path, "run-b", model_id="model-b", pricing=run_b_pricing))
    judge = FakeJudge(usage=usage) if usage else FakeJudge()
    judgement = await judge_runs(a, b, judge, "judge-x", 0, pricing=pricing, progress=lambda _m: None)
    return a, b, judgement


class TestSummaryJson:
    async def test_judge_usage_and_total_cost_alongside_run_cost(self, tmp_path):
        a, b, judgement = await _judged(tmp_path)
        summary = build_summary([a, b], judgement)
        json.dumps(summary)

        assert summary["judgement"]["judge_usage"] == {
            "judge_model": "judge-x", "input_tokens": 15_000, "output_tokens": 500,
            "cost_usd": pytest.approx(JUDGE_COST), "calls_without_usage": 0, "prices_as_of": "2026-03-01",
        }
        assert summary["costs"]["runs_usd"] == pytest.approx(2 * RUN_COST)
        assert summary["costs"]["judge_usd"] == pytest.approx(JUDGE_COST)
        assert summary["costs"]["total_usd"] == pytest.approx(2 * RUN_COST + JUDGE_COST)
        assert summary["costs"]["judge_included"] is True
        assert summary["prices_as_of"] == ["2026-01-15", "2026-03-01"]

    async def test_digest_prints_judge_cost_and_the_total(self, tmp_path):
        a, b, judgement = await _judged(tmp_path)
        text = summary_text(build_summary([a, b], judgement))
        assert "judging [judge-x]: cost $0.0700 (15000 in / 500 out tokens)" in text
        assert "runs + judging total $0.1100" in text

    async def test_an_unpriced_judge_shows_unknown_and_the_total_is_unknown_not_the_runs_alone(self, tmp_path):
        a, b, judgement = await _judged(tmp_path, pricing=UNPRICED)
        summary = build_summary([a, b], judgement)
        assert summary["costs"]["judge_usd"] is None and summary["costs"]["total_usd"] is None
        assert summary["costs"]["runs_usd"] == pytest.approx(2 * RUN_COST)
        assert summary["judgement"]["judge_usage"]["input_tokens"] == 15_000
        text = summary_text(summary)
        assert "judging [judge-x]: cost unknown" in text
        assert "runs + judging total unknown" in text

    async def test_a_judgement_without_recorded_usage_says_so(self, tmp_path):
        a, b, judgement = await _judged(tmp_path, usage=None)
        summary = build_summary([a, b], judgement)
        usage = summary["judgement"]["judge_usage"]
        assert usage["cost_usd"] is None and usage["input_tokens"] is None
        assert "judging [judge-x]: cost unknown (usage not recorded)" in summary_text(summary)

    async def test_an_unknown_run_cost_makes_the_total_unknown_even_with_a_known_judge_cost(self, tmp_path):
        a, b, judgement = await _judged(tmp_path, run_b_pricing=UNPRICED)
        costs = build_summary([a, b], judgement)["costs"]
        assert costs["runs_usd"] is None and costs["total_usd"] is None
        assert costs["judge_usd"] == pytest.approx(JUDGE_COST)

    def test_without_a_judgement_the_total_is_the_runs_alone(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a"))
        summary = build_summary([run])
        assert summary["costs"] == {
            "runs_usd": pytest.approx(RUN_COST), "judge_usd": None, "judge_included": False,
            "total_usd": pytest.approx(RUN_COST),
        }
        assert "judging" not in summary_text(summary)

    async def test_summarize_cli_writes_the_costs_block(self, tmp_path, capsys):
        a, b, judgement = await _judged(tmp_path)
        write_model(tmp_path / "judged" / "judgement.json", judgement)
        out = tmp_path / "summary.json"
        code = main(["summarize", "--runs", str(a.dir), str(b.dir), "--judgement", str(tmp_path / "judged"), "--out", str(out)])
        assert code == 0
        assert json.loads(out.read_text())["costs"]["total_usd"] == pytest.approx(2 * RUN_COST + JUDGE_COST)
        assert "runs + judging total $0.1100" in capsys.readouterr().out


class TestReportPdf:
    async def test_the_pdf_has_a_judge_cost_line_and_the_total(self, tmp_path):
        a, b, judgement = await _judged(tmp_path)
        text = pdf_text_blob(render_report([a, b], judgement, build_summary([a, b], judgement)))
        assert "Judge cost: $0.0700 (15000 in / 500 out tokens)." in text
        assert "Run cost: $0.0400. Runs + judging total: $0.1100." in text

    async def test_the_pdf_shows_an_unpriced_judge_as_unknown(self, tmp_path):
        a, b, judgement = await _judged(tmp_path, pricing=UNPRICED)
        text = pdf_text_blob(render_report([a, b], judgement, build_summary([a, b], judgement)))
        assert "Judge cost: unknown (15000 in / 500 out tokens)." in text
        assert "Runs + judging total: unknown." in text


async def _comparison(tmp_path, *, pricing=JUDGE_PRICING):
    ref = load_run(write_run(tmp_path, "ref", model_id="ref-m"))
    model = load_run(write_run(tmp_path, "mod", model_id="mod-m"))
    judgement = await judge_runs(
        model, ref, FakeJudge(usage=(3000, 100)), "judge-x", 0, pricing=pricing, progress=lambda _m: None
    )
    summary = build_comparison_summary(
        ref, [model], {"mod-m": judgement}, graph="arch-x", judge_model="judge-x",
        run_dirs={"ref-m": "runs/ref-m", "mod-m": "runs/mod-m"},
    )
    return summary, {"ref-m": ref, "mod-m": model}, {"mod-m": judgement}


class TestComparisonSummaryAndPdf:
    async def test_total_is_every_run_plus_all_judging(self, tmp_path):
        summary, _, _ = await _comparison(tmp_path)
        json.dumps(summary)
        assert summary["costs"]["runs_usd"] == pytest.approx(2 * RUN_COST)
        assert summary["costs"]["judge_usd"] == pytest.approx(JUDGE_COST)
        assert summary["costs"]["total_usd"] == pytest.approx(2 * RUN_COST + JUDGE_COST)
        assert summary["models"]["ref-m"]["judge"] is None
        judge = summary["models"]["mod-m"]["judge"]
        assert judge["judge_model"] == "judge-x" and judge["cost_usd"] == pytest.approx(JUDGE_COST)
        assert judge["input_tokens"] == 15_000
        assert summary["prices_as_of"] == ["2026-01-15", "2026-03-01"]

    async def test_an_unpriced_judge_makes_the_total_unknown(self, tmp_path):
        summary, _, _ = await _comparison(tmp_path, pricing=UNPRICED)
        assert summary["costs"]["judge_usd"] is None and summary["costs"]["total_usd"] is None
        assert summary["costs"]["runs_usd"] == pytest.approx(2 * RUN_COST)

    async def test_terminal_digest_shows_runs_judging_and_total(self, tmp_path):
        summary, _, _ = await _comparison(tmp_path)
        text = comparison_text(summary)
        assert "Cost: runs $0.0400 + judging $0.0700 = total $0.1100" in text
        assert "judging $0.0700" in text.split("Recommendations:")[0].splitlines()[2]  # on the model's line

    async def test_pdf_has_the_judge_cost_line_with_the_total(self, tmp_path):
        summary, runs, judgements = await _comparison(tmp_path)
        text = pdf_text_blob(render_comparison_report(summary, runs, judgements))
        assert "Run cost: $0.0400. Judge cost: $0.0700. Total (runs + judging): $0.1100." in text

    async def test_pdf_shows_unknown_for_an_unpriced_judge(self, tmp_path):
        summary, runs, judgements = await _comparison(tmp_path, pricing=UNPRICED)
        text = pdf_text_blob(render_comparison_report(summary, runs, judgements))
        assert "Judge cost: unknown. Total (runs + judging): unknown." in text



def _partial_usage(prompt: str):
    """Usage on every judge call except the one about q-risk (that call reports none)."""
    return None if "(q-risk)" in prompt else (3000, 100)


class TestPartialJudgeUsage:
    """Token totals that miss some calls are a lower bound, and every report says so."""

    CAVEAT = "12000 in / 400 out tokens (some calls reported no usage)"

    async def test_summary_digest_and_pdf_flag_a_lower_bound(self, tmp_path):
        a, b, judgement = await _judged(tmp_path, usage=_partial_usage)
        assert judgement.judge_calls_without_usage == 1
        summary = build_summary([a, b], judgement)
        assert self.CAVEAT in summary_text(summary)
        assert self.CAVEAT in pdf_text_blob(render_report([a, b], judgement, summary))

    async def test_the_agreement_digest_flags_a_lower_bound(self, tmp_path):
        from evals.benchmark.agreement import agreement_text, compare_judgements

        _, _, first = await _judged(tmp_path, usage=_partial_usage)
        assert self.CAVEAT in agreement_text(compare_judgements(first, first))
