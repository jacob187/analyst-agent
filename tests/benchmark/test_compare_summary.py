"""AC3 (per-model JSON summary, comparison PDF) and the recommendation rules, on fixture runs.

Runs are written with the real schema; judgements come from the real judge driven by the
`FakeJudge` (the LLM boundary). Nothing here calls a model or the network.
"""

import json

import pytest

from evals.benchmark.compare_report import render_comparison_report, representative_ids
from evals.benchmark.compare_summary import (
    MIN_WIN_TIE_RATE,
    build_comparison_summary,
    comparison_text,
    vs_reference,
)
from evals.benchmark.judge import judge_runs
from evals.benchmark.runs import load_run
from evals.benchmark.schemas import Pricing
from tests.benchmark.helpers import FIXTURE_QUESTIONS, PRICED, UNPRICED, FakeJudge, pdf_text_blob, write_run

pytestmark = pytest.mark.eval_unit

IDS = [q[0] for q in FIXTURE_QUESTIONS]
CATEGORIES = {q[2] for q in FIXTURE_QUESTIONS}

# Per question, with 1000 input / 200 output tokens (see helpers.make_result).
CHEAPEST = Pricing(input_price_per_mtok=0.1, output_price_per_mtok=0.5, prices_as_of="2026-01-15")  # 0.0002
CHEAP = Pricing(input_price_per_mtok=0.2, output_price_per_mtok=1.0, prices_as_of="2026-01-15")  # 0.0004
PRICEY = Pricing(input_price_per_mtok=200.0, output_price_per_mtok=1000.0, prices_as_of="2026-01-15")  # 0.4
# PRICED (the reference by default) costs 0.004 per question.


def good(*qids: str) -> dict[str, str]:
    return {qid: f"GOOD answer for {qid}" for qid in qids}


def _raise(_text):
    raise RuntimeError("judge exploded")


async def build(tmp_path, challengers, *, reference=None, decide=None, multiple=3.0):
    """challengers: {model_id: write_run kwargs}. Returns (summary, runs, judgements)."""
    reference = reference or {}
    ref = load_run(write_run(tmp_path, "ref", model_id="ref-m", **{"pricing": PRICED, **reference}))
    runs = {"ref-m": ref}
    judgements = {}
    for model_id, kwargs in challengers.items():
        run = load_run(write_run(tmp_path, model_id, model_id=model_id, **kwargs))
        runs[model_id] = run
        judgements[model_id] = await judge_runs(
            run, ref, FakeJudge(decide), "judge-x", 0, progress=lambda _m: None
        )
    summary = build_comparison_summary(
        ref, [r for m, r in runs.items() if m != "ref-m"], judgements,
        graph="arch-x", judge_model="judge-x",
        run_dirs={m: f"runs/{m}" for m in runs}, chat_cost_multiple=multiple,
    )
    return summary, runs, judgements


class TestSummaryShape:
    async def test_per_model_metrics_against_the_reference(self, tmp_path):
        summary, _, _ = await build(tmp_path, {
            "cheap-m": dict(pricing=CHEAP, answers=good(*IDS)),
            "pricey-m": dict(pricing=PRICEY, answers=good("q-risk", "q-multi", "q-8k")),
        })
        json.dumps(summary)  # serialisable

        assert summary["reference"] == "ref-m"
        assert list(summary["models"]) == ["ref-m", "cheap-m", "pricey-m"]
        assert summary["prices_as_of"] == ["2026-01-15"]
        assert summary["judge_model"] == "judge-x" and summary["graph"] == "arch-x"
        assert set(summary["recommendations"]) == {"chat_default", "free_trial", "filing_analysis"}
        assert set(summary["recommendation_rules"]) >= {"chat_default", "free_trial", "filing_analysis"}

        ref, cheap, pricey = (summary["models"][m] for m in ("ref-m", "cheap-m", "pricey-m"))
        assert ref["is_reference"] and ref["vs_reference"] is None
        assert ref["means"]["cost_usd"] == pytest.approx(0.004)
        assert ref["prices_as_of"] == "2026-01-15"

        vs = cheap["vs_reference"]
        assert (vs["wins"], vs["ties"], vs["losses"], vs["judged"]) == (5, 0, 0, 5)
        assert (vs["win_rate"], vs["win_or_tie_rate"], vs["win_score"]) == (1.0, 1.0, 1.0)
        assert (vs["unique_factual_errors"], vs["reference_unique_factual_errors"]) == (0, 0)
        assert cheap["means"]["cost_usd"] == pytest.approx(0.0004)
        assert cheap["means"]["llm_calls"] == 3 and cheap["means"]["total_tokens"] == 1200
        assert cheap["means"]["latency_s"] == 10
        assert cheap["pricing"] == CHEAP.model_dump()

        vs = pricey["vs_reference"]
        assert (vs["wins"], vs["ties"], vs["losses"]) == (3, 2, 0)
        assert vs["win_score"] == pytest.approx(0.8)
        assert pricey["means"]["cost_usd"] == pytest.approx(0.4)

    async def test_per_category_breakdown_has_means_and_the_outcome(self, tmp_path):
        summary, _, _ = await build(tmp_path, {
            "cheap-m": dict(pricing=CHEAP, answers=good("q-risk")),
        })
        by_cat = summary["models"]["cheap-m"]["by_category"]
        assert set(by_cat) == CATEGORIES
        filing = by_cat["filing_lookup"]
        assert filing["n"] == 1
        assert filing["means"]["cost_usd"] == pytest.approx(0.0004)
        assert (filing["vs_reference"]["wins"], filing["vs_reference"]["ties"]) == (1, 0)
        assert by_cat["technicals"]["vs_reference"]["ties"] == 1
        assert summary["models"]["ref-m"]["by_category"]["technicals"]["vs_reference"] is None

    async def test_unique_error_counts_for_both_sides(self, tmp_path):
        summary, _, _ = await build(
            tmp_path,
            {"wrong-m": dict(pricing=CHEAP, answers={"q-risk": "WRONG claim", "q-tech": "WRONG again"})},
            reference={"answers": {"q-news": "WRONG number"}},
        )
        vs = summary["models"]["wrong-m"]["vs_reference"]
        assert vs["unique_factual_errors"] == 2
        assert vs["reference_unique_factual_errors"] == 1

    async def test_a_question_the_model_failed_to_answer_counts_as_a_loss(self, tmp_path):
        summary, runs, judgements = await build(tmp_path, {"drop-m": dict(pricing=CHEAP, answers={"q-tech": None})})
        vs = summary["models"]["drop-m"]["vs_reference"]
        assert (vs["judged"], vs["unanswered_by_model"], vs["ties"]) == (4, 1, 4)
        assert vs["win_or_tie_rate"] == pytest.approx(0.8)  # 4 of 5, not 4 of 4
        assert vs_reference(runs["drop-m"], runs["ref-m"], None) is None

        # A question the reference also lacks is skipped without blaming the model.
        ref2 = load_run(write_run(tmp_path, "ref2", model_id="ref2-m", answers={"q-tech": None}))
        judgement = await judge_runs(runs["drop-m"], ref2, FakeJudge(), "judge-x", 0, progress=lambda _m: None)
        both = vs_reference(runs["drop-m"], ref2, judgement)
        assert (both["unanswered_by_model"], both["other_skipped"], both["judged"]) == (0, 1, 4)
        ref3 = load_run(write_run(tmp_path, "ref3", model_id="ref3-m", answers={"q-news": None}))
        judgement = await judge_runs(runs["drop-m"], ref3, FakeJudge(), "judge-x", 0, progress=lambda _m: None)
        mixed = vs_reference(runs["drop-m"], ref3, judgement)
        assert (mixed["unanswered_by_model"], mixed["other_skipped"], mixed["judged"]) == (1, 1, 3)

    async def test_unknown_prices_stay_unknown(self, tmp_path):
        summary, _, _ = await build(tmp_path, {"unp-m": dict(pricing=UNPRICED)})
        block = summary["models"]["unp-m"]
        assert block["means"]["cost_usd"] is None
        assert block["cost_unknown_questions"] == 5
        assert block["prices_as_of"] is None
        assert "unknown" in comparison_text(summary)

    async def test_text_digest_prints_recommendations_and_the_rules(self, tmp_path):
        summary, _, _ = await build(tmp_path, {"cheap-m": dict(pricing=CHEAP, answers=good(*IDS))})
        text = comparison_text(summary)
        assert "free_trial: cheap-m" in text
        assert "Rules:" in text and f"{MIN_WIN_TIE_RATE:.0%}" in text
        assert "prices_as_of: 2026-01-15" in text


class TestRecommendations:
    async def test_free_trial_is_the_cheapest_model_that_keeps_quality(self, tmp_path):
        summary, _, _ = await build(tmp_path, {
            "cheap-m": dict(pricing=CHEAP, answers=good(*IDS)),
            "pricey-m": dict(pricing=PRICEY, answers=good(*IDS)),
        })
        rec = summary["recommendations"]["free_trial"]
        assert rec["model_id"] == "cheap-m"
        assert "$0.0004" in rec["rationale"]
        assert rec["rule"]

    async def test_a_cheaper_model_with_a_unique_factual_error_is_not_eligible(self, tmp_path):
        summary, _, _ = await build(tmp_path, {
            "wrong-m": dict(pricing=CHEAPEST, answers={"q-risk": "WRONG claim"}),  # ties otherwise
            "cheap-m": dict(pricing=CHEAP, answers=good(*IDS)),
        })
        assert summary["models"]["wrong-m"]["means"]["cost_usd"] < summary["models"]["cheap-m"]["means"]["cost_usd"]
        assert summary["recommendations"]["free_trial"]["model_id"] == "cheap-m"

    async def test_a_cheaper_model_below_the_win_tie_threshold_is_not_eligible(self, tmp_path):
        summary, _, _ = await build(tmp_path, {
            # One failed answer of five: 4/5 = 80% win+tie, under the 85% bar.
            "drop-m": dict(pricing=CHEAPEST, answers={"q-tech": None}),
            "cheap-m": dict(pricing=CHEAP, answers=good(*IDS)),
        })
        assert summary["recommendations"]["free_trial"]["model_id"] == "cheap-m"

    async def test_free_trial_can_stay_with_the_reference(self, tmp_path):
        summary, _, _ = await build(tmp_path, {"pricey-m": dict(pricing=PRICEY, answers=good(*IDS))})
        assert summary["recommendations"]["free_trial"]["model_id"] == "ref-m"

    async def test_chat_default_is_the_best_model_within_the_cost_multiple(self, tmp_path):
        challengers = {
            "cheap-m": dict(pricing=CHEAP, answers=good("q-risk", "q-multi", "q-8k")),  # win score 0.8
            "pricey-m": dict(pricing=PRICEY, answers=good(*IDS)),  # win score 1.0, 100x the reference
        }
        within, _, _ = await build(tmp_path / "a", challengers, multiple=3.0)
        assert within["recommendations"]["chat_default"]["model_id"] == "cheap-m"
        assert "3x" in within["recommendations"]["chat_default"]["rationale"]
        assert within["recommendation_rules"]["chat_default"].count("3x") == 1

        generous, _, _ = await build(tmp_path / "b", challengers, multiple=1000.0)
        assert generous["recommendations"]["chat_default"]["model_id"] == "pricey-m"

    async def test_chat_default_stays_with_the_reference_when_nothing_beats_it(self, tmp_path):
        summary, _, _ = await build(
            tmp_path,
            {"weak-m": dict(pricing=CHEAP)},  # ties everywhere except where the reference is GOOD
            reference={"answers": good("q-risk", "q-multi")},
        )
        assert summary["models"]["weak-m"]["vs_reference"]["losses"] == 2
        assert summary["recommendations"]["chat_default"]["model_id"] == "ref-m"

    async def test_filing_analysis_uses_the_filing_lookup_category_only(self, tmp_path):
        summary, _, _ = await build(
            tmp_path,
            {
                "filing-m": dict(pricing=CHEAP, answers=good("q-risk")),  # wins filing_lookup, loses q-multi
                "other-m": dict(pricing=CHEAP, answers=good("q-tech", "q-news", "q-8k")),
            },
            reference={"answers": good("q-multi")},
        )
        rec = summary["recommendations"]["filing_analysis"]
        assert rec["model_id"] == "filing-m"
        assert "filing_lookup" in rec["rationale"]

    async def test_models_with_judge_errors_are_excluded_and_named(self, tmp_path):
        summary, _, _ = await build(tmp_path, {"cheap-m": dict(pricing=CHEAP, answers=good(*IDS))}, decide=_raise)
        vs = summary["models"]["cheap-m"]["vs_reference"]
        assert vs["judge_errors"] == 5 and vs["judged"] == 0 and vs["win_score"] is None
        for key in ("free_trial", "chat_default", "filing_analysis"):
            rec = summary["recommendations"][key]
            assert rec["model_id"] == "ref-m"  # the only model with usable data
            assert "cheap-m (" in rec["rationale"] and "judge errors" in rec["rationale"]

    async def test_models_with_unknown_cost_are_excluded_and_named(self, tmp_path):
        summary, _, _ = await build(tmp_path, {"unp-m": dict(pricing=UNPRICED, answers=good(*IDS))})
        rec = summary["recommendations"]["free_trial"]
        assert rec["model_id"] == "ref-m"
        assert "unp-m (unknown cost)" in rec["rationale"]

    async def test_unknown_reference_cost_makes_the_cost_cap_rules_insufficient(self, tmp_path):
        summary, _, _ = await build(
            tmp_path, {"cheap-m": dict(pricing=CHEAP, answers=good(*IDS))}, reference={"pricing": UNPRICED}
        )
        for key in ("chat_default", "filing_analysis"):
            rec = summary["recommendations"][key]
            assert rec["model_id"] is None
            assert rec["rationale"].startswith("Insufficient data:")
            assert "reference" in rec["rationale"]
        # free_trial needs no cap, so the model with a known cost can still qualify.
        assert summary["recommendations"]["free_trial"]["model_id"] == "cheap-m"

    async def test_no_known_cost_anywhere_is_insufficient_not_a_guess(self, tmp_path):
        summary, _, _ = await build(
            tmp_path, {"unp-m": dict(pricing=UNPRICED, answers=good(*IDS))}, reference={"pricing": UNPRICED}
        )
        for key, rec in summary["recommendations"].items():
            assert rec["model_id"] is None, key
            assert rec["rationale"].startswith("Insufficient data:"), key
        assert "none - Insufficient data" in comparison_text(summary)


class TestComparisonReport:
    @pytest.fixture
    async def fixture(self, tmp_path):
        return await build(tmp_path, {
            "cheap-m": dict(pricing=CHEAP, answers=good(*IDS)),
            "wrong-m": dict(pricing=UNPRICED, answers={"q-tech": None, "q-news": "WRONG number"}),
            "pricey-m": dict(pricing=PRICEY),
        })

    async def test_pdf_is_valid_and_has_every_question_and_model_id(self, fixture):
        summary, runs, judgements = fixture
        pdf = render_comparison_report(summary, runs, judgements)
        assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
        text = pdf_text_blob(pdf)
        for qid in IDS:
            assert qid in text, qid
        for model_id in summary["models"]:
            assert model_id in text, model_id

    async def test_pdf_has_the_table_recommendations_and_prices(self, fixture):
        summary, runs, judgements = fixture
        text = pdf_text_blob(render_comparison_report(summary, runs, judgements))
        assert text.index("Per-model summary") < text.index("Recommendations") < text.index("Appendix A")
        for key in ("chat_default", "free_trial", "filing_analysis"):
            assert key in text
        assert "2026-01-15" in text  # prices_as_of
        assert "unknown" in text  # the unpriced model
        assert "$0.0004" in text  # cheap-m cost per question
        assert "(no answer)" in text  # wrong-m's failed question

    async def test_answers_are_shown_side_by_side_for_the_models(self, fixture):
        summary, runs, judgements = fixture
        text = pdf_text_blob(render_comparison_report(summary, runs, judgements))
        assert "GOOD answer for q-risk" in text
        assert "pricey-m answer for q-risk" in text and "ref answer for q-risk" in text

    async def test_more_models_than_one_row_still_render(self, tmp_path):
        challengers = {f"m{i}-m": dict(pricing=CHEAP) for i in range(5)}
        summary, runs, judgements = await build(tmp_path, challengers)
        text = pdf_text_blob(render_comparison_report(summary, runs, judgements))
        assert all(f"m{i}-m answer for q-risk" in text for i in range(5))


class TestRepresentativeSubset:
    def test_first_question_of_each_category_in_question_set_order(self, tmp_path):
        run = load_run(write_run(tmp_path, "r"))
        results = dict(run.questions)
        ids = ["q-news", "q-risk", "q-multi", "q-tech", "q-8k"]
        assert representative_ids({"question_ids": ids}, results) == ids
        # Two questions in one category: only the first is featured.
        results["q-risk2"] = results["q-risk"].model_copy(update={"id": "q-risk2"})
        picked = representative_ids({"question_ids": ["q-risk2", *ids]}, results)
        assert picked.count("q-risk2") == 1 and "q-risk" not in picked
        assert len(picked) == len(CATEGORIES)
