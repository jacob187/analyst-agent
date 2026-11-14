"""AC1: the judge's own token usage and cost are measured per question and in total.

The only fake is the chat model at the LLM boundary (`FakeJudge(usage=...)`: a real langchain
`with_structured_output` pipeline over a scripted chat model), so the usage comes from genuine
OpenInference spans via `capture_spans`.
"""

import json

import pytest

from agents.model_registry import get_free_tier_model
from evals.benchmark.cli import main
from evals.benchmark.common import provider_env_var
from evals.benchmark.judge import JudgeVerdict, judge_runs
from evals.benchmark.runs import load_judgement, load_run, write_model
from evals.benchmark.schemas import Judgement
from tests.benchmark.helpers import FIXTURE_QUESTIONS, PRICED, UNPRICED, FakeJudge, write_run

pytestmark = pytest.mark.eval_unit

IDS = [q[0] for q in FIXTURE_QUESTIONS]


@pytest.fixture
def runs(tmp_path):
    a = write_run(tmp_path, "run-a", answers={qid: f"plain {qid}" for qid in IDS})
    b = write_run(tmp_path, "run-b", answers={qid: f"GOOD {qid}" for qid in IDS})
    return load_run(a), load_run(b)


async def _judge(runs, judge, pricing=PRICED):
    return await judge_runs(runs[0], runs[1], judge, "judge-model-x", 0, pricing=pricing, progress=lambda _m: None)


class TestJudgeUsageIsRecorded:
    async def test_per_question_and_total_tokens_and_cost(self, runs):
        calls = []

        def usage(prompt):  # a different usage per call, so per-question figures are told apart
            calls.append(prompt)
            return 1000 * len(calls), 100 * len(calls)

        judgement = await _judge(runs, FakeJudge(usage=usage))

        assert [q.status for q in judgement.questions] == ["judged"] * 5
        assert [(q.judge_input_tokens, q.judge_output_tokens) for q in judgement.questions] == [
            (1000, 100), (2000, 200), (3000, 300), (4000, 400), (5000, 500),
        ]
        # PRICED: $2 / Mtok in, $10 / Mtok out
        assert judgement.questions[1].judge_cost_usd == pytest.approx((2000 * 2 + 200 * 10) / 1e6)
        assert judgement.judge_input_tokens == 15_000
        assert judgement.judge_output_tokens == 1_500
        assert judgement.judge_cost_usd == pytest.approx((15_000 * 2 + 1_500 * 10) / 1e6)
        assert judgement.judge_cost_usd == pytest.approx(sum(q.judge_cost_usd for q in judgement.questions))
        assert judgement.judge_prices_as_of == "2026-01-15"
        assert judgement.judge_calls_without_usage == 0

    async def test_the_structured_verdict_is_unchanged_by_the_span_capture(self, runs):
        with_usage = await _judge(runs, FakeJudge(usage=(10, 1)))
        without = await _judge(runs, FakeJudge())
        assert [(q.winner, q.a_unique_factual_error, q.rationale) for q in with_usage.questions] == [
            (q.winner, q.a_unique_factual_error, q.rationale) for q in without.questions
        ]

    async def test_an_unpriced_judge_model_has_tokens_but_unknown_cost(self, runs):
        judgement = await _judge(runs, FakeJudge(usage=(1000, 100)), pricing=UNPRICED)
        assert judgement.judge_input_tokens == 5000 and judgement.judge_output_tokens == 500
        assert judgement.judge_cost_usd is None
        assert all(q.judge_cost_usd is None and q.judge_input_tokens == 1000 for q in judgement.questions)
        assert judgement.judge_prices_as_of is None

    async def test_no_pricing_passed_is_also_unknown_not_zero(self, runs):
        judgement = await _judge(runs, FakeJudge(usage=(1000, 100)), pricing=None)
        assert judgement.judge_cost_usd is None
        assert judgement.judge_input_tokens == 5000

    async def test_a_judge_that_reports_no_usage_is_unknown_not_zero(self, runs):
        judgement = await _judge(runs, FakeJudge())  # no LLM span at all
        assert judgement.judge_cost_usd is None
        assert judgement.judge_input_tokens is None and judgement.judge_output_tokens is None
        assert all(q.judge_input_tokens is None and q.judge_cost_usd is None for q in judgement.questions)
        assert judgement.judge_calls_without_usage == 5

    async def test_a_call_whose_reply_has_no_usage_metadata_makes_the_cost_unknown(self, runs):
        n = []

        def usage(_prompt):
            n.append(1)
            return None if len(n) == 3 else (1000, 100)

        judgement = await _judge(runs, FakeJudge(usage=usage))
        assert judgement.questions[2].judge_input_tokens is None
        assert judgement.questions[2].judge_cost_usd is None
        assert judgement.questions[0].judge_cost_usd is not None
        assert judgement.judge_input_tokens == 4000  # a lower bound
        assert judgement.judge_cost_usd is None
        assert judgement.judge_calls_without_usage == 1

    async def test_a_call_that_fails_to_parse_still_records_what_it_spent(self, runs):
        def decide(text):
            return {"winner": "banana"} if len(judge.prompts) == 2 else _good(text)

        judge = FakeJudge(usage=(1000, 100), decide=decide)
        judgement = await _judge(runs, judge)
        assert judgement.questions[1].status == "error"
        assert judgement.questions[1].judge_input_tokens == 1000
        assert judgement.judge_input_tokens == 5000

    async def test_skipped_questions_cost_nothing_and_are_not_counted(self, tmp_path):
        a = write_run(tmp_path, "run-a", answers={"q-tech": None})
        b = write_run(tmp_path, "run-b")
        judgement = await _judge((load_run(a), load_run(b)), FakeJudge(usage=(1000, 100)))
        skipped = next(q for q in judgement.questions if q.status == "skipped")
        assert skipped.judge_input_tokens is None and skipped.judge_cost_usd is None
        assert judgement.judge_input_tokens == 4000
        assert judgement.judge_calls_without_usage == 0


def _good(text):
    return JudgeVerdict(
        winner="tie", first_has_unique_factual_error=False, second_has_unique_factual_error=False, rationale="ok"
    )


class TestOlderJudgementsStillLoad:
    def test_a_judgement_file_without_any_cost_fields_loads_with_none(self, tmp_path):
        old = {
            "judge_model": "judge-model-x",
            "seed": 1,
            "date": "2026-02-02T00:00:00+00:00",
            "question_set_hash": "h",
            "a": {"run_id": "ra", "dir": "a", "architecture": "x", "model_id": "m"},
            "b": {"run_id": "rb", "dir": "b", "architecture": "x", "model_id": "m"},
            "questions": [
                {"id": "q1", "category": "c", "ticker": "AAPL", "status": "judged", "winner": "a", "rationale": "r"}
            ],
        }
        (tmp_path / "judgement.json").write_text(json.dumps(old))
        judgement = load_judgement(tmp_path)
        assert judgement.judge_input_tokens is None and judgement.judge_output_tokens is None
        assert judgement.judge_cost_usd is None and judgement.judge_prices_as_of is None
        assert judgement.judge_calls_without_usage is None
        assert judgement.questions[0].judge_cost_usd is None

    def test_cost_fields_round_trip_through_the_file(self, tmp_path):
        judgement = Judgement.model_validate(
            {
                "judge_model": "j", "seed": 1, "date": "d", "question_set_hash": "h",
                "a": {"run_id": "ra", "dir": "a", "architecture": "x", "model_id": "m"},
                "b": {"run_id": "rb", "dir": "b", "architecture": "x", "model_id": "m"},
                "questions": [],
                "judge_input_tokens": 10, "judge_output_tokens": 2, "judge_cost_usd": 0.5,
                "judge_prices_as_of": "2026-01-15",
            }
        )
        write_model(tmp_path / "judgement.json", judgement)
        assert load_judgement(tmp_path) == judgement


class TestJudgeCli:
    def test_judge_command_prices_the_judge_with_the_registry_model_and_prints_its_cost(
        self, monkeypatch, tmp_path, capsys
    ):
        model = get_free_tier_model()
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.setattr("evals.benchmark.cli.prepare_environment", lambda: None)
        monkeypatch.setenv(provider_env_var(model), "fake-key-value-0123")
        monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec, **kw: FakeJudge(usage=(1000, 100)))
        write_run(tmp_path, "run-a", answers={"q-risk": "plain"})
        write_run(tmp_path, "run-b", answers={"q-risk": "GOOD"})

        code = main(["judge", "--a", str(tmp_path / "run-a"), "--b", str(tmp_path / "run-b"),
                     "--seed", "0", "--out", str(tmp_path / "judged")])

        assert code == 0
        judgement = load_judgement(tmp_path / "judged")
        expected = (5 * 1000 * model.input_price_per_mtok + 5 * 100 * model.output_price_per_mtok) / 1e6
        assert judgement.judge_cost_usd == pytest.approx(expected)
        assert judgement.judge_prices_as_of == str(model.prices_as_of)
        assert f"Judge usage: 5000 in / 500 out tokens, cost ${expected:.4f}" in capsys.readouterr().out
