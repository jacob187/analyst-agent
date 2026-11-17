"""AC4: `compare-judgements` on two judgements of the same run pair.

The judgements are built directly with hand-picked verdicts so every expected number is
computed by hand (see FIRST / SECOND below); nothing calls a judge or the network.
"""

import json

import pytest

from evals.benchmark.agreement import (
    AGREEMENT_THRESHOLD,
    RATIONALE_LIMIT,
    agreement_text,
    cohens_kappa,
    compare_judgements,
)
from evals.benchmark.cli import main
from evals.benchmark.common import BenchmarkError
from evals.benchmark.runs import write_model
from evals.benchmark.schemas import Judgement, JudgedQuestion, RunRef

pytestmark = pytest.mark.eval_unit

LONG = "x" * 300

# id: (category, winner, a_unique_error, b_unique_error, rationale); status "judged" unless noted.
FIRST = {
    "q1": ("filing_lookup", "a", False, False, "first q1"),
    "q2": ("filing_lookup", "a", False, False, LONG),
    "q3": ("multi_step", "b", True, False, "first q3"),
    "q4": ("multi_step", "tie", False, False, "first q4"),
    "q5": ("technicals", "b", False, True, "first q5"),
    "q6": ("technicals", "a", False, False, "first q6"),
    "q7": ("market_news", "a", False, False, "first q7"),  # the second judge errored here
    "q8": ("market_news", None, None, None, None),  # skipped by both
}
SECOND = {
    "q1": ("filing_lookup", "a", False, False, "second q1"),
    "q2": ("filing_lookup", "b", False, False, "second q2"),
    "q3": ("multi_step", "b", False, False, "second q3"),
    "q4": ("multi_step", "tie", False, False, "second q4"),
    "q5": ("technicals", "b", False, True, "second q5"),
    "q6": ("technicals", "tie", False, False, "second q6"),
    "q7": ("market_news", None, None, None, None),
    "q8": ("market_news", None, None, None, None),
}
STATUS_OVERRIDE_SECOND = {"q7": "error", "q8": "skipped"}
STATUS_OVERRIDE_FIRST = {"q8": "skipped"}


def judgement(rows, *, judge_model, cost=None, tokens=(None, None), a="run-a", b="run-b", statuses=None):
    questions = []
    for qid, (category, winner, a_err, b_err, rationale) in rows.items():
        status = (statuses or {}).get(qid, "judged")
        questions.append(
            JudgedQuestion(
                id=qid, category=category, ticker="AAPL", status=status,
                winner=winner if status == "judged" else None,
                a_unique_factual_error=a_err if status == "judged" else None,
                b_unique_factual_error=b_err if status == "judged" else None,
                rationale=rationale if status == "judged" else None,
                reason=None if status == "judged" else status,
            )
        )
    return Judgement(
        judge_model=judge_model, seed=1, date="2026-02-02T00:00:00+00:00", question_set_hash="h",
        a=RunRef(run_id=a, dir="a", architecture="arch", model_id="m-a"),
        b=RunRef(run_id=b, dir="b", architecture="arch", model_id="m-b"),
        questions=questions,
        judge_cost_usd=cost, judge_input_tokens=tokens[0], judge_output_tokens=tokens[1],
    )


@pytest.fixture
def first():
    return judgement(FIRST, judge_model="expensive-judge", cost=0.5, tokens=(9000, 900), statuses=STATUS_OVERRIDE_FIRST)


@pytest.fixture
def second():
    return judgement(SECOND, judge_model="cheap-judge", cost=0.05, tokens=(9000, 800), statuses=STATUS_OVERRIDE_SECOND)


class TestAgreementNumbers:
    def test_winner_agreement_counts_only_questions_both_judged(self, first, second):
        report = compare_judgements(first, second)
        # q1..q6 judged by both; the winners agree on q1, q3, q4, q5.
        assert report["compared"] == 6
        assert report["winner_agreed"] == 4
        assert report["winner_agreement"] == pytest.approx(0.6667, abs=1e-4)
        assert report["judged_by_only_one"] == ["q7"]  # q8 was skipped by both: not counted anywhere

    def test_each_unique_error_flag_is_compared_separately(self, first, second):
        report = compare_judgements(first, second)
        a_flag, b_flag = report["a_unique_factual_error"], report["b_unique_factual_error"]
        # Run A's flag differs only on q3 (first flagged it, second did not).
        assert (a_flag["agreed"], a_flag["agreement"]) == (5, pytest.approx(0.8333, abs=1e-4))
        assert (a_flag["both_flagged"], a_flag["either_flagged"]) == (0, 1)
        # Run B's flag agrees everywhere; both flagged q5.
        assert (b_flag["agreed"], b_flag["agreement"]) == (6, 1.0)
        assert (b_flag["both_flagged"], b_flag["either_flagged"]) == (1, 1)

    def test_cohens_kappa_for_the_winner(self, first, second):
        # first: a,a,b,tie,b,a  second: a,b,b,tie,b,tie
        # p_o = 4/6; p_e = (3/6)(1/6) + (2/6)(3/6) + (1/6)(2/6) = 11/36; kappa = (24/36 - 11/36) / (25/36) = 0.52
        assert compare_judgements(first, second)["winner_kappa"] == pytest.approx(0.52)

    def test_per_category_agreement(self, first, second):
        by_category = compare_judgements(first, second)["by_category"]
        assert set(by_category) == {"filing_lookup", "multi_step", "technicals"}
        assert (by_category["filing_lookup"]["compared"], by_category["filing_lookup"]["winner_agreed"]) == (2, 1)
        assert by_category["filing_lookup"]["winner_agreement"] == 0.5
        assert by_category["multi_step"]["winner_agreement"] == 1.0
        assert by_category["multi_step"]["a_unique_factual_error"]["agreed"] == 1  # q3 differs, q4 agrees
        assert by_category["technicals"]["winner_agreement"] == 0.5

    def test_disagreements_list_both_winners_and_clipped_rationales(self, first, second):
        disagreements = compare_judgements(first, second)["disagreements"]
        assert [(d["id"], d["differs_on"]) for d in disagreements] == [
            ("q2", ["winner"]), ("q3", ["a_unique_factual_error"]), ("q6", ["winner"]),
        ]
        q2 = disagreements[0]
        assert q2["category"] == "filing_lookup"
        assert (q2["first"]["winner"], q2["second"]["winner"]) == ("a", "b")
        assert q2["second"]["rationale"] == "second q2"
        assert len(q2["first"]["rationale"]) == RATIONALE_LIMIT
        assert q2["first"]["rationale"] == "x" * (RATIONALE_LIMIT - 3) + "..."
        q6 = disagreements[2]
        assert (q6["first"]["winner"], q6["second"]["winner"]) == ("a", "tie")

    def test_each_judge_model_and_recorded_cost_is_reported(self, first, second):
        report = compare_judgements(first, second)
        assert report["first_judge"]["judge_model"] == "expensive-judge"
        assert report["first_judge"]["cost_usd"] == 0.5
        assert report["second_judge"]["judge_model"] == "cheap-judge"
        assert (report["second_judge"]["cost_usd"], report["second_judge"]["output_tokens"]) == (0.05, 800)

    def test_the_85_percent_rule_is_guidance_and_evaluated_on_all_three_rates(self, first, second):
        guidance = compare_judgements(first, second)["guidance"]
        assert guidance["threshold"] == AGREEMENT_THRESHOLD == 0.85
        assert "agreement >= 85% on winners and on error flags" in guidance["text"]
        assert "cheaper judge is acceptable" in guidance["text"]
        assert guidance["meets"] is False  # winners agree on 67%

    def test_identical_judgements_meet_the_guidance(self, first):
        report = compare_judgements(first, first.model_copy(deep=True))
        assert report["winner_agreement"] == 1.0 and report["disagreements"] == []
        assert report["guidance"]["meets"] is True


class TestSameRunPair:
    def test_a_judgement_of_different_runs_is_refused(self, first):
        other = judgement(FIRST, judge_model="j", b="run-c", statuses=STATUS_OVERRIDE_FIRST)
        with pytest.raises(BenchmarkError, match="not of the same run pair"):
            compare_judgements(first, other)

    def test_only_one_matching_run_is_still_refused(self, first):
        other = judgement(FIRST, judge_model="j", a="run-z", statuses=STATUS_OVERRIDE_FIRST)
        with pytest.raises(BenchmarkError, match="not of the same run pair"):
            compare_judgements(first, other)

    def test_a_pair_judged_in_the_opposite_order_is_re_oriented_not_inverted(self, first, second):
        flip = {"a": "b", "b": "a", "tie": "tie", None: None}
        swapped_rows = {
            qid: (cat, flip[w], b_err, a_err, why) for qid, (cat, w, a_err, b_err, why) in SECOND.items()
        }
        swapped = judgement(swapped_rows, judge_model="cheap-judge", a="run-b", b="run-a", statuses=STATUS_OVERRIDE_SECOND)
        straight = compare_judgements(first, second)
        flipped = compare_judgements(first, swapped)
        assert flipped["winner_agreement"] == straight["winner_agreement"]
        assert flipped["a_unique_factual_error"] == straight["a_unique_factual_error"]
        assert flipped["winner_kappa"] == straight["winner_kappa"]
        assert flipped["disagreements"] == straight["disagreements"]


class TestEdges:
    def test_no_question_judged_by_both_gives_none_not_zero(self, first):
        nothing = judgement(FIRST, judge_model="j", statuses={q: "error" for q in FIRST})
        report = compare_judgements(first, nothing)
        assert report["compared"] == 0
        assert report["winner_agreement"] is None and report["winner_kappa"] is None
        assert report["guidance"]["meets"] is None
        assert "undefined" in agreement_text(report)

    def test_kappa_is_undefined_when_chance_agreement_is_certain(self):
        assert cohens_kappa(["a", "a"], ["a", "a"]) is None
        assert cohens_kappa([], []) is None

    def test_judge_cost_unknown_is_shown_as_unknown(self, second):
        unpriced = judgement(FIRST, judge_model="expensive-judge", cost=None, tokens=(9000, 900), statuses=STATUS_OVERRIDE_FIRST)
        text = agreement_text(compare_judgements(unpriced, second))
        assert "expensive-judge, judge cost unknown (9000 in / 900 out tokens)" in text
        assert "cheap-judge, judge cost $0.0500 (9000 in / 800 out tokens)" in text

    def test_an_old_judgement_without_usage_says_so(self, first):
        old = judgement(SECOND, judge_model="cheap-judge", statuses=STATUS_OVERRIDE_SECOND)
        assert "judge cost unknown (usage not recorded)" in agreement_text(compare_judgements(first, old))


class TestDigest:
    def test_digest_has_the_rates_disagreements_and_the_guidance(self, first, second):
        text = agreement_text(compare_judgements(first, second))
        assert "Compared 6 question(s) judged by both judges; 1 judged by only one" in text
        assert "Winner agreement: 67% (4/6); Cohen's kappa: 0.52" in text
        assert "Unique-factual-error flag agreement, run A: 83% (5/6; flagged by both 0, by either 1)" in text
        assert "Unique-factual-error flag agreement, run B: 100% (6/6; flagged by both 1, by either 1)" in text
        assert "filing_lookup: 50% (1/2)" in text
        assert "q2 [filing_lookup] differs on winner: first a / second b" in text
        assert "Guidance, not a gate: agreement >= 85% on winners and on error flags -> the cheaper judge is acceptable." in text
        assert "below the threshold on at least one" in text


class TestCli:
    def _write(self, tmp_path, first, second):
        write_model(tmp_path / "judged-1" / "judgement.json", first)
        write_model(tmp_path / "judged-2" / "judgement.json", second)
        return ["compare-judgements", "--a", str(tmp_path / "judged-1"), "--b", str(tmp_path / "judged-2")]

    def test_prints_the_digest_and_writes_the_report_json(self, tmp_path, first, second, capsys):
        args = self._write(tmp_path, first, second)
        out = tmp_path / "nested" / "agreement.json"
        assert main([*args, "--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "Winner agreement: 67% (4/6)" in printed and f"Wrote {out}" in printed
        report = json.loads(out.read_text())
        assert report["winner_agreed"] == 4 and report["guidance"]["meets"] is False
        assert [d["id"] for d in report["disagreements"]] == ["q2", "q3", "q6"]

    def test_low_agreement_is_not_an_error_exit(self, tmp_path, first, second):
        assert main(self._write(tmp_path, first, second)) == 0  # guidance only, never a gate

    def test_different_runs_exit_2_with_a_clear_error(self, tmp_path, first, capsys):
        other = judgement(FIRST, judge_model="j", b="run-c", statuses=STATUS_OVERRIDE_FIRST)
        assert main(self._write(tmp_path, first, other)) == 2
        err = capsys.readouterr().err
        assert "not of the same run pair" in err and "run-c" in err

    def test_a_missing_judgement_directory_exits_2(self, tmp_path, first, capsys):
        write_model(tmp_path / "judged-1" / "judgement.json", first)
        assert main(["compare-judgements", "--a", str(tmp_path / "judged-1"), "--b", str(tmp_path / "nope")]) == 2
        assert "Cannot read" in capsys.readouterr().err
