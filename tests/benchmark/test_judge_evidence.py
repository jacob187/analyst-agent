"""AC2 + AC3: the judge knows today's date and sees each run's evidence; evidence trimming is bounded,
deterministic and recorded.

The only fake is the judge LLM at the boundary (`with_structured_output(...).ainvoke`).
"""

import json
from datetime import datetime, timezone

import pytest

from evals.benchmark.judge import (
    EVIDENCE_BUDGET_CHARS,
    MIN_ITEM_CHARS,
    RenderedEvidence,
    judge_runs,
    render_evidence,
)
from evals.benchmark.runs import load_judgement, load_run
from evals.benchmark.schemas import ToolEvidence
from evals.benchmark.summary import build_summary
from tests.benchmark.helpers import FIXTURE_QUESTIONS, FakeJudge, evidence_block, write_run

pytestmark = pytest.mark.eval_unit

IDS = [q[0] for q in FIXTURE_QUESTIONS]
SEED = 0  # mixes A-first and B-first orders across the five fixture questions
SECRETS = ("arch-alpha", "arch-beta", "model-alpha", "model-beta", "run-a", "run-b", "judge-model-x")

ALPHA_EVIDENCE = [
    ToolEvidence(tool="get_risk_factors", input="{'ticker': 'AAPL'}", output="EVIDENCE-ALPHA revenue $94.9B"),
]
BETA_EVIDENCE = [
    ToolEvidence(tool="get_stock_price", input="{}", output="EVIDENCE-BETA close 227.50", truncated=True),
    ToolEvidence(tool="web_search", input="{'query': 'news'}", output="EVIDENCE-BETA headline", error=True),
]


async def _judge(runs, judge=None):
    judge = judge or FakeJudge()
    judgement = await judge_runs(runs[0], runs[1], judge, "judge-model-x", SEED, progress=lambda _m: None)
    return judgement, judge


@pytest.fixture
def evidence_runs(tmp_path):
    answers = {qid: f"answer text for {qid}" for qid in IDS}  # the default answers embed the run name
    a = write_run(tmp_path, "run-a", architecture="arch-alpha", model_id="model-alpha", answers=answers,
                  date="2026-02-01T10:00:00+00:00", evidence=ALPHA_EVIDENCE)
    b = write_run(tmp_path, "run-b", architecture="arch-beta", model_id="model-beta", answers=answers,
                  date="2026-03-09T18:30:00+00:00", evidence=BETA_EVIDENCE)
    return load_run(a), load_run(b)


class TestPromptDateAndEvidence:
    async def test_prompt_states_the_date_and_that_newer_than_the_judges_knowledge_is_not_a_hallucination(
        self, evidence_runs
    ):
        judgement, judge = await _judge(evidence_runs)
        assert len(judge.prompts) == len(IDS)
        for prompt in judge.prompts:
            assert "Today's date: 2026-03-09" in prompt  # the later run's date
            assert "newer than anything you know" in prompt
            assert "is NOT a" in prompt and "hallucination" in prompt
        assert judgement.as_of_date == "2026-03-09"

    async def test_each_side_gets_its_own_runs_evidence_in_the_position_that_run_was_shown(self, evidence_runs):
        judgement, judge = await _judge(evidence_runs)
        assert {q.a_is_first for q in judgement.questions} == {True, False}  # both orders exercised
        for q, prompt in zip(judgement.questions, judge.prompts):
            first, second = evidence_block(prompt, "FIRST"), evidence_block(prompt, "SECOND")
            a_block, b_block = (first, second) if q.a_is_first else (second, first)
            assert "EVIDENCE-ALPHA" in a_block and "EVIDENCE-BETA" not in a_block
            assert "EVIDENCE-BETA" in b_block and "EVIDENCE-ALPHA" not in b_block
            assert "tool: get_risk_factors" in a_block and "input: {'ticker': 'AAPL'}" in a_block
            assert "tool: get_stock_price" in b_block
            assert "[output was cut when the run was recorded]" in b_block
            assert "tool: web_search (failed)" in b_block

    async def test_the_prompt_with_evidence_is_still_blind(self, evidence_runs):
        _, judge = await _judge(evidence_runs)
        for prompt in judge.prompts:
            for secret in SECRETS:
                assert secret not in prompt
            assert "FIRST'S EVIDENCE" in prompt and "SECOND'S EVIDENCE" in prompt

    def test_factual_error_is_defined_against_the_evidence(self):
        from agents.prompts import BENCHMARK_JUDGE_SYSTEM_PROMPT as system

        assert "contradicts" in system and "no provided evidence supports" in system

    async def test_runs_without_recorded_evidence_are_marked_and_still_judged(self, tmp_path):
        a, b = write_run(tmp_path, "run-a"), write_run(tmp_path, "run-b")  # no evidence field: older runs
        judgement, judge = await _judge((load_run(a), load_run(b)))
        for prompt in judge.prompts:
            assert all("Evidence was not recorded" in evidence_block(prompt, side) for side in ("FIRST", "SECOND"))
        assert all(
            q.status == "judged" and q.a_evidence_calls is None and q.b_evidence_calls is None
            for q in judgement.questions
        )

    async def test_a_run_that_made_no_evidence_bearing_calls_is_told_apart_from_unrecorded(self, tmp_path):
        a = write_run(tmp_path, "run-a", evidence=[])
        b = write_run(tmp_path, "run-b", evidence=ALPHA_EVIDENCE)
        judgement, judge = await _judge((load_run(a), load_run(b)))
        first = judgement.questions[0]
        assert (first.a_evidence_calls, first.b_evidence_calls) == (0, 1)
        texts = {evidence_block(judge.prompts[0], "FIRST"), evidence_block(judge.prompts[0], "SECOND")}
        assert any("No tool outputs were retrieved" in t for t in texts)

    async def test_the_date_falls_back_to_judge_time_when_run_dates_do_not_parse(self, tmp_path):
        a = write_run(tmp_path, "run-a", date="not-a-date")
        b = write_run(tmp_path, "run-b", date="also not")
        judgement, judge = await _judge((load_run(a), load_run(b)))
        assert judgement.as_of_date == datetime.now(timezone.utc).date().isoformat()
        assert f"Today's date: {judgement.as_of_date}" in judge.prompts[0]

    async def test_evidence_issues_note_is_kept_and_blank_becomes_none(self, evidence_runs):
        def decide(text):
            return {
                "winner": "tie", "first_has_unique_factual_error": False, "second_has_unique_factual_error": False,
                "evidence_issues": "  claims $9B, evidence says $8B  " if "tool: get_stock_price" in text else "",
                "rationale": "r",
            }

        judgement, _ = await _judge(evidence_runs, judge=FakeJudge(decide=decide))
        assert {q.evidence_issues for q in judgement.questions} == {"claims $9B, evidence says $8B"}

        quiet, _ = await _judge(evidence_runs)  # the fixture judge never sets evidence_issues
        assert {q.evidence_issues for q in quiet.questions} == {None}


def _item(i: int, size: int) -> ToolEvidence:
    return ToolEvidence(tool=f"tool_{i}", input="{}", output=f"{i}" * size)


class TestEvidenceTrimming:
    def test_evidence_within_budget_is_untouched(self):
        rendered = render_evidence([_item(1, 1000), _item(2, 2000)])
        assert rendered.trimmed is False and rendered.calls == 2
        assert "1" * 1000 in rendered.text and "2" * 2000 in rendered.text
        assert "cut by the judge" not in rendered.text

    def test_over_budget_evidence_is_cut_to_the_budget_and_flagged(self):
        rendered = render_evidence([_item(1, 100_000), _item(2, 100_000)])
        assert rendered.trimmed is True and rendered.calls == 2
        # Headers count against the budget; only the blank-line separators come on top.
        assert len(rendered.text) <= EVIDENCE_BUDGET_CHARS + 2 * (rendered.calls - 1)
        assert "tool_1" in rendered.text and "tool_2" in rendered.text  # every call stays represented
        assert rendered.text.count("cut by the judge's evidence budget") == 2

    def test_trimming_is_deterministic(self):
        items = [_item(i, 30_000 + i) for i in range(5)]
        assert render_evidence(items) == render_evidence(list(items))

    def test_short_calls_keep_all_their_text_and_long_ones_share_the_rest_equally(self):
        text = render_evidence([_item(2, 60_000), _item(1, 300), _item(3, 60_000)]).text
        assert "1" * 300 in text  # the small call is whole
        assert "cut by the judge" in text
        kept_2, kept_3 = text.count("2"), text.count("3")
        assert abs(kept_2 - kept_3) <= 2 and kept_2 > 15_000
        assert text.index("tool_2") < text.index("tool_1") < text.index("tool_3")  # call order is preserved

    def test_the_head_of_each_over_share_call_is_what_survives(self):
        text = render_evidence([ToolEvidence(tool="t", output="HEAD" + "z" * 100_000 + "TAIL")]).text
        assert "HEAD" in text and "TAIL" not in text

    def test_too_many_calls_keeps_the_earliest_and_counts_the_rest(self):
        rendered = render_evidence([_item(i, 2_000) for i in range(200)])
        kept = EVIDENCE_BUDGET_CHARS // MIN_ITEM_CHARS
        assert rendered.calls == 200 and rendered.trimmed is True
        assert f"tool: tool_{kept - 1}\n" in rendered.text and f"tool: tool_{kept}\n" not in rendered.text
        assert f"{200 - kept} further tool call(s) omitted" in rendered.text
        assert len(rendered.text) <= EVIDENCE_BUDGET_CHARS + 100 * kept

    def test_unrecorded_and_empty_evidence(self):
        assert render_evidence(None) == RenderedEvidence("(Evidence was not recorded for this answer.)", None, False)
        assert render_evidence([]).calls == 0

    async def test_trimming_is_recorded_per_run_in_the_judgement_and_the_prompt_is_bounded(self, tmp_path):
        a = write_run(tmp_path, "run-a", evidence=[_item(1, 100_000)])
        b = write_run(tmp_path, "run-b", evidence=[_item(2, 500)])
        judgement, judge = await _judge((load_run(a), load_run(b)))
        assert judgement.evidence_budget_chars == EVIDENCE_BUDGET_CHARS
        for q, prompt in zip(judgement.questions, judge.prompts):
            assert q.a_evidence_trimmed is True and q.b_evidence_trimmed is False
            assert q.a_evidence_calls == q.b_evidence_calls == 1
            assert len(prompt) < 2 * EVIDENCE_BUDGET_CHARS

    async def test_judging_twice_produces_identical_prompts(self, tmp_path):
        a = write_run(tmp_path, "run-a", evidence=[_item(1, 100_000), _item(2, 90_000)])
        b = write_run(tmp_path, "run-b", evidence=[_item(3, 100_000)])
        _, first = await _judge((load_run(a), load_run(b)))
        _, again = await _judge((load_run(a), load_run(b)))
        assert first.prompts == again.prompts


class TestJudgementFilesAndSummary:
    async def test_a_judgement_without_the_new_fields_loads_and_summarises(self, tmp_path):
        a, b = load_run(write_run(tmp_path, "run-a")), load_run(write_run(tmp_path, "run-b"))
        old = {
            "judge_model": "judge-model-x", "seed": 1, "date": "2026-01-01T00:00:00+00:00",
            "question_set_hash": a.meta.question_set_hash,
            "a": {"run_id": a.meta.run_id, "dir": "a", "architecture": "x", "model_id": "m"},
            "b": {"run_id": b.meta.run_id, "dir": "b", "architecture": "y", "model_id": "n"},
            "questions": [
                {"id": "q-risk", "category": "filing_lookup", "ticker": "AAPL", "status": "judged",
                 "a_is_first": True, "winner": "a", "a_unique_factual_error": False,
                 "b_unique_factual_error": True, "rationale": "r", "raw_verdict": {"winner": "first"}}
            ],
        }
        out = tmp_path / "old-judgement"
        out.mkdir()
        (out / "judgement.json").write_text(json.dumps(old))
        judgement = load_judgement(out)
        assert judgement.as_of_date is None and judgement.questions[0].a_evidence_trimmed is None
        block = build_summary([a, b], judgement)["judgement"]
        assert block["as_of_date"] is None and block["evidence_trimmed_questions"] == 0
        assert block["evidence_issues"] == []

    async def test_summary_reports_date_budget_trimming_and_issue_notes(self, tmp_path):
        a = write_run(tmp_path, "run-a", evidence=[_item(1, 100_000)])
        b = write_run(tmp_path, "run-b", evidence=[_item(2, 10)])
        runs = (load_run(a), load_run(b))
        judge = FakeJudge(decide=lambda text: {
            "winner": "tie", "first_has_unique_factual_error": False, "second_has_unique_factual_error": False,
            "evidence_issues": "figure not in evidence", "rationale": "r",
        })
        judgement, _ = await _judge(runs, judge=judge)
        block = build_summary(list(runs), judgement)["judgement"]
        assert block["as_of_date"] == "2026-02-01"
        assert block["evidence_budget_chars"] == EVIDENCE_BUDGET_CHARS
        assert block["evidence_trimmed_questions"] == len(IDS)
        assert {i["id"] for i in block["evidence_issues"]} == set(IDS)
