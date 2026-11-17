"""AC4 (summarize writes JSON) and AC7 (cost from price snapshot, prices_as_of, unknown)."""

import json

import pytest

from evals.benchmark.cli import main
from evals.benchmark.judge import judge_runs
from evals.benchmark.runs import load_run, write_model
from evals.benchmark.summary import build_summary, summary_text
from tests.benchmark.helpers import FIXTURE_QUESTIONS, PRICED, UNPRICED, FakeJudge, pdf_text_blob, write_run

pytestmark = pytest.mark.eval_unit

IDS = [q[0] for q in FIXTURE_QUESTIONS]


def _by_label(summary):
    return {r["label"]: r for r in summary["runs"]}


class TestBuildSummary:
    def test_totals_means_and_cost_per_run(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a"))
        (block,) = build_summary([run])["runs"]

        assert block["questions"] == {"selected": 5, "recorded": 5, "answered": 5, "failed": 0, "with_errors": 0}
        assert block["totals"]["llm_calls"] == 15
        assert block["means"]["llm_calls"] == 3
        assert block["totals"]["input_tokens"] == 5000 and block["totals"]["output_tokens"] == 1000
        assert block["means"]["total_tokens"] == 1200
        assert block["totals"]["latency_s"] == 50
        assert block["means"]["latency_s"] == 10
        # (1000 * 2 + 200 * 10) / 1e6 = 0.004 per question
        assert block["totals"]["cost_usd"] == pytest.approx(0.02)
        assert block["means"]["cost_usd"] == pytest.approx(0.004)
        assert block["cost_unknown_questions"] == 0
        assert block["pricing"] == PRICED.model_dump()

    def test_per_category_breakdown(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a"))
        (block,) = build_summary([run])["runs"]
        assert set(block["by_category"]) == {"filing_lookup", "multi_step", "material_events", "technicals", "market_news"}
        assert block["by_category"]["technicals"]["n"] == 1
        assert block["by_category"]["technicals"]["totals"]["cost_usd"] == pytest.approx(0.004)

    def test_missing_prices_make_cost_unknown_never_zero(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a", pricing=UNPRICED))
        summary = build_summary([run])
        (block,) = summary["runs"]
        assert block["totals"]["cost_usd"] is None
        assert block["means"]["cost_usd"] is None
        assert block["cost_unknown_questions"] == 5
        assert block["cost_known_usd"] == 0
        assert block["by_category"]["technicals"]["totals"]["cost_usd"] is None
        assert summary["prices_as_of"] == []
        assert "unknown" in summary_text(summary)
        assert "no price data" in summary_text(summary)

    def test_a_call_without_token_usage_makes_that_run_s_cost_unknown(self, tmp_path):
        run_dir = write_run(tmp_path, "run-a")
        run = load_run(run_dir)
        run.questions["q-tech"] = run.questions["q-tech"].model_copy(update={"llm_calls_without_usage": 1})
        (block,) = build_summary([run])["runs"]
        assert block["totals"]["cost_usd"] is None
        assert block["cost_unknown_questions"] == 1
        assert block["cost_known_usd"] == pytest.approx(0.016)  # the four fully-observed questions
        assert block["llm_calls_without_usage"] == 1

    def test_prices_as_of_dates_are_collected_across_runs(self, tmp_path):
        other = PRICED.model_copy(update={"prices_as_of": "2026-02-20"})
        runs = [
            load_run(write_run(tmp_path, "run-a")),
            load_run(write_run(tmp_path, "run-b", pricing=other)),
            load_run(write_run(tmp_path, "run-c", pricing=UNPRICED)),
        ]
        summary = build_summary(runs)
        assert summary["prices_as_of"] == ["2026-01-15", "2026-02-20"]
        text = summary_text(summary)
        assert "prices_as_of: 2026-01-15, 2026-02-20" in text
        assert _by_label(summary)["run-c"]["totals"]["cost_usd"] is None

    def test_failed_questions_and_errors_are_counted(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a", answers={"q-tech": None}, errors=["tool X failed"]))
        (block,) = build_summary([run])["runs"]
        assert block["questions"]["failed"] == 1
        assert block["questions"]["answered"] == 4
        assert block["questions"]["with_errors"] == 5

    def test_mismatched_question_sets_and_tavily_settings_warn(self, tmp_path):
        a = load_run(write_run(tmp_path, "run-a"))
        b = load_run(write_run(tmp_path, "run-b", question_set_hash="other", tavily_enabled=False))
        warnings = " ".join(build_summary([a, b])["warnings"])
        assert "different question sets" in warnings
        assert "Tavily" in warnings


class TestJudgementInSummary:
    async def test_win_tie_loss_and_unique_error_counts(self, tmp_path):
        answers_a = {qid: f"plain {qid}" for qid in IDS}
        answers_b = {qid: f"GOOD {qid}" for qid in IDS}
        answers_a["q-risk"] = "plain WRONG"
        answers_a["q-news"] = answers_b["q-news"] = "GOOD same"
        a = load_run(write_run(tmp_path, "run-a", answers=answers_a))
        b = load_run(write_run(tmp_path, "run-b", answers=answers_b))
        judgement = await judge_runs(a, b, FakeJudge(), "judge-model-x", 0, progress=lambda _m: None)

        summary = build_summary([a, b], judgement)
        runs = _by_label(summary)
        assert runs["run-a"]["judgement"] == {
            "judged": 5, "skipped_or_errored": 0, "wins": 0, "losses": 4, "ties": 1,
            "unique_factual_errors": 1, "opponent_unique_factual_errors": 0,
        }
        assert runs["run-b"]["judgement"]["wins"] == 4
        assert runs["run-b"]["judgement"]["losses"] == 0
        assert runs["run-b"]["judgement"]["opponent_unique_factual_errors"] == 1
        block = summary["judgement"]
        assert block["judge_model"] == "judge-model-x" and block["seed"] == 0
        assert block["a_counts"]["wins"] == 0 and block["b_counts"]["wins"] == 4
        assert block["by_category"]["filing_lookup"] == {"a_wins": 0, "b_wins": 1, "ties": 0}
        assert block["by_category"]["market_news"] == {"a_wins": 0, "b_wins": 0, "ties": 1}

    async def test_a_judgement_for_other_runs_is_flagged(self, tmp_path):
        a, b = (load_run(write_run(tmp_path, n)) for n in ("run-a", "run-b"))
        c = load_run(write_run(tmp_path, "run-c"))
        judgement = await judge_runs(a, b, FakeJudge(), "j", 0, progress=lambda _m: None)
        summary = build_summary([c], judgement)
        assert summary["runs"][0]["judgement"] is None
        assert any("does not match" in w for w in summary["warnings"])


class TestSummarizeCli:
    async def test_writes_json_and_prints_prices_as_of(self, tmp_path, capsys):
        a = write_run(tmp_path, "run-a")
        b = write_run(tmp_path, "run-b", pricing=UNPRICED)
        judgement = await judge_runs(
            load_run(a), load_run(b), FakeJudge(), "judge-model-x", 0, progress=lambda _m: None
        )
        write_model(tmp_path / "judged" / "judgement.json", judgement)
        out = tmp_path / "out" / "summary.json"

        code = main(["summarize", "--runs", str(a), str(b), "--judgement", str(tmp_path / "judged"), "--out", str(out)])

        assert code == 0
        data = json.loads(out.read_text())
        assert [r["label"] for r in data["runs"]] == ["run-a", "run-b"]
        assert data["prices_as_of"] == ["2026-01-15"]
        assert data["runs"][1]["totals"]["cost_usd"] is None
        assert data["judgement"]["a"]["architecture"] == "arch-x"
        printed = capsys.readouterr().out
        assert "prices_as_of: 2026-01-15" in printed
        assert "total cost unknown" in printed

    def test_a_missing_run_dir_is_a_clean_error(self, tmp_path, capsys):
        code = main(["summarize", "--runs", str(tmp_path / "nope"), "--out", str(tmp_path / "s.json")])
        assert code == 2
        assert "Cannot read" in capsys.readouterr().err


class TestDeltasAndFailingIds:
    """Spec 08 AC4/AC5: the old-vs-new change per question, and which questions failed in each run."""

    HEAVY = dict(llm_calls=6, input_tokens=2000, output_tokens=400, total_tokens=2400, latency_s=20.0)

    def test_two_runs_get_absolute_and_percentage_deltas_old_to_new(self, tmp_path):
        old = load_run(write_run(tmp_path, "old"))
        new = load_run(write_run(tmp_path, "new", **self.HEAVY))
        deltas = build_summary([old, new])["deltas"]

        assert (deltas["old"], deltas["new"]) == ("old", "new")
        calls = deltas["means"]["llm_calls"]
        assert (calls["old"], calls["new"], calls["delta"], calls["pct"]) == (3, 6, 3, 100.0)
        assert deltas["means"]["input_tokens"]["delta"] == 1000
        assert deltas["means"]["output_tokens"]["delta"] == 200
        assert deltas["means"]["latency_s"]["delta"] == 10 and deltas["means"]["latency_s"]["pct"] == 100.0
        cost = deltas["means"]["cost_usd"]  # 0.004 -> 0.008 per question
        assert cost["delta"] == pytest.approx(0.004) and cost["pct"] == 100.0
        assert deltas["totals"]["cost_usd"]["delta"] == pytest.approx(0.02)

    def test_a_decrease_is_negative(self, tmp_path):
        old = load_run(write_run(tmp_path, "old", **self.HEAVY))
        new = load_run(write_run(tmp_path, "new"))
        calls = build_summary([old, new])["deltas"]["means"]["llm_calls"]
        assert (calls["delta"], calls["pct"]) == (-3, -50.0)

    def test_an_unknown_cost_gives_no_cost_delta_never_zero(self, tmp_path):
        old = load_run(write_run(tmp_path, "old", pricing=UNPRICED))
        new = load_run(write_run(tmp_path, "new"))
        summary = build_summary([old, new])
        cost = summary["deltas"]["means"]["cost_usd"]
        assert cost["old"] is None and cost["delta"] is None and cost["pct"] is None
        assert "cost change unknown" in summary_text(summary)

    def test_a_zero_baseline_has_a_delta_but_no_percentage(self, tmp_path):
        old = load_run(write_run(tmp_path, "old", llm_calls=0))
        new = load_run(write_run(tmp_path, "new"))
        calls = build_summary([old, new])["deltas"]["means"]["llm_calls"]
        assert calls["delta"] == 3 and calls["pct"] is None

    def test_deltas_exist_only_for_exactly_two_runs(self, tmp_path):
        a, b, c = (load_run(write_run(tmp_path, n)) for n in ("a", "b", "c"))
        assert build_summary([a])["deltas"] is None
        assert build_summary([a, b, c])["deltas"] is None

    def test_failed_and_errored_question_ids_are_listed_per_run(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a", answers={"q-tech": None, "q-news": None}))
        run.questions["q-risk"] = run.questions["q-risk"].model_copy(update={"errors": ["tool X failed"]})
        (block,) = build_summary([run])["runs"]
        assert block["failed_ids"] == ["q-news", "q-tech"]
        assert block["error_ids"] == ["q-risk"]  # answered, but a tool errored along the way

    def test_a_clean_run_lists_no_ids(self, tmp_path):
        (block,) = build_summary([load_run(write_run(tmp_path, "run-a"))])["runs"]
        assert block["failed_ids"] == [] and block["error_ids"] == []

    def test_the_text_digest_prints_the_deltas_and_the_failing_ids(self, tmp_path):
        old = load_run(write_run(tmp_path, "old"))
        new = load_run(write_run(tmp_path, "new", answers={"q-tech": None}, **self.HEAVY))
        text = summary_text(build_summary([old, new]))
        assert "old -> new" in text and "LLM calls +3.0 (+100.0%)" in text
        assert "cost +$0.0040 (+100.0%)" in text
        assert "new: failed q-tech" in text
        assert "old: failed none" in text

    def test_the_cli_summary_json_and_the_pdf_carry_them(self, tmp_path, capsys):
        a = write_run(tmp_path, "run-a")
        b = write_run(tmp_path, "run-b", answers={"q-tech": None}, **self.HEAVY)
        out = tmp_path / "s.json"
        pdf = tmp_path / "r.pdf"
        assert main(["summarize", "--runs", str(a), str(b), "--out", str(out)]) == 0
        assert "run-a -> run-b" in capsys.readouterr().out
        assert main(["report", "--runs", str(a), str(b), "--out", str(pdf)]) == 0

        data = json.loads(out.read_text())
        assert data["deltas"]["means"]["llm_calls"]["delta"] == 3
        assert data["runs"][1]["failed_ids"] == ["q-tech"]
        text = pdf_text_blob(pdf.read_bytes())
        assert "Change run-a -> run-b" in text and "Failed questions" in text and "q-tech" in text
