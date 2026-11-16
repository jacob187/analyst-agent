"""AC3: blind pairwise judging, recorded shuffle, un-shuffle, structured-output parsing.

The only fake is the judge LLM at the boundary (`with_structured_output(...).ainvoke`).
"""

from types import SimpleNamespace

import pytest

from evals.benchmark.common import BenchmarkError
from evals.benchmark.judge import JudgeVerdict, a_is_first, judge_runs
from evals.benchmark.runs import load_run
from tests.benchmark.helpers import FIXTURE_QUESTIONS, PRICED, FakeJudge, block, write_run

pytestmark = pytest.mark.eval_unit

IDS = [q[0] for q in FIXTURE_QUESTIONS]
# With seed 0 the fixture questions get a mix of A-first and B-first orders.
SEED = 0


@pytest.fixture
def runs(tmp_path):
    answers_a = {qid: f"plain answer {qid}" for qid in IDS}
    answers_b = {qid: f"GOOD answer {qid}" for qid in IDS}
    answers_a["q-risk"] = "plain WRONG claim"          # A has a unique error here
    answers_b["q-8k"] = "GOOD WRONG claim"              # B has a unique error here (and still wins)
    answers_a["q-news"] = answers_b["q-news"] = "GOOD identical"  # tie
    a = write_run(tmp_path, "run-a", architecture="arch-alpha", model_id="model-alpha", answers=answers_a)
    b = write_run(tmp_path, "run-b", architecture="arch-beta", model_id="model-beta", answers=answers_b)
    return load_run(a), load_run(b)


async def _judge(runs, judge=None, seed=SEED):
    judge = judge or FakeJudge()
    judgement = await judge_runs(runs[0], runs[1], judge, "judge-model-x", seed, progress=lambda _m: None)
    return judgement, judge


class TestShuffleAndUnshuffle:
    async def test_winner_and_error_flags_map_back_to_runs_regardless_of_order(self, runs):
        judgement, judge = await _judge(runs)
        by_id = {q.id: q for q in judgement.questions}

        orders = {q.a_is_first for q in judgement.questions}
        assert orders == {True, False}  # the fixture really exercises both orders

        assert by_id["q-risk"].winner == "b"
        assert by_id["q-risk"].a_unique_factual_error is True
        assert by_id["q-risk"].b_unique_factual_error is False
        assert by_id["q-8k"].winner == "b"
        assert by_id["q-8k"].a_unique_factual_error is False
        assert by_id["q-8k"].b_unique_factual_error is True
        assert by_id["q-news"].winner == "tie"
        for qid in ("q-multi", "q-tech"):
            assert by_id[qid].winner == "b"
            assert by_id[qid].a_unique_factual_error is by_id[qid].b_unique_factual_error is False
        assert all(q.status == "judged" and q.rationale for q in judgement.questions)
        assert JudgeVerdict in judge.schemas

    async def test_order_is_recorded_and_matches_what_the_judge_saw(self, runs):
        judgement, judge = await _judge(runs)
        assert judgement.seed == SEED
        for q, prompt in zip(judgement.questions, judge.prompts):
            assert q.a_is_first == a_is_first(SEED, q.id)
            if q.id != "q-news":  # identical answers there; elsewhere A starts "plain", B "GOOD"
                assert block(prompt, "FIRST").startswith("plain" if q.a_is_first else "GOOD")
                assert block(prompt, "SECOND").startswith("GOOD" if q.a_is_first else "plain")

    async def test_same_seed_replays_the_same_order_and_seeds_differ(self, runs):
        first, _ = await _judge(runs, seed=SEED)
        again, _ = await _judge(runs, seed=SEED)
        other, _ = await _judge(runs, seed=3)
        assert [q.a_is_first for q in first.questions] == [q.a_is_first for q in again.questions]
        assert [q.a_is_first for q in first.questions] != [q.a_is_first for q in other.questions]

    async def test_raw_verdict_is_kept_in_the_judges_frame(self, runs):
        judgement, _ = await _judge(runs)
        for q in judgement.questions:
            assert q.raw_verdict["winner"] in {"first", "second", "tie"}


class TestBlindness:
    async def test_prompts_reveal_neither_architecture_model_nor_run_label(self, runs):
        _, judge = await _judge(runs)
        assert len(judge.prompts) == len(IDS)
        for prompt in judge.prompts:
            for secret in ("arch-alpha", "arch-beta", "model-alpha", "model-beta", "run-a", "run-b", "judge-model-x"):
                assert secret not in prompt

    def test_prompts_are_loaded_from_agents_prompts_with_system_and_user_templates(self):
        from agents import prompts

        user = prompts.BENCHMARK_JUDGE_USER_PROMPT
        assert "{first_answer}" in user and "{second_answer}" in user
        assert prompts.BENCHMARK_JUDGE_SYSTEM_PROMPT.strip()


class TestRobustness:
    async def test_questions_without_an_answer_in_either_run_are_skipped_with_a_reason(self, tmp_path):
        a = write_run(tmp_path, "run-a", answers={"q-tech": None})
        b = write_run(tmp_path, "run-b", answers={"q-news": None})
        judgement, judge = await _judge((load_run(a), load_run(b)))
        by_id = {q.id: q for q in judgement.questions}
        assert by_id["q-tech"].status == by_id["q-news"].status == "skipped"
        assert "run A" in by_id["q-tech"].reason and "run B" in by_id["q-news"].reason
        assert len(judge.prompts) == len(IDS) - 2  # the judge was not asked about them

    async def test_a_judge_failure_is_recorded_and_the_rest_continue(self, runs):
        calls = []

        def flaky(text):
            calls.append(text)
            if len(calls) == 2:
                raise RuntimeError("provider 500")
            return {
                "winner": "tie",
                "first_has_unique_factual_error": False,
                "second_has_unique_factual_error": False,
                "rationale": "dict verdicts are accepted too",
            }

        judgement, _ = await _judge(runs, judge=FakeJudge(decide=flaky))
        statuses = [q.status for q in judgement.questions]
        assert statuses == ["judged", "error", "judged", "judged", "judged"]
        assert "provider 500" in judgement.questions[1].reason

    @pytest.mark.parametrize("wrapper_key", ["JudgeVerdict", "verdict"])
    async def test_a_verdict_wrapped_in_a_single_key_is_unwrapped(self, runs, wrapper_key):
        def wrapped(text):
            first, second = block(text, "FIRST"), block(text, "SECOND")
            winner = "first" if "GOOD" in first and "GOOD" not in second else (
                "second" if "GOOD" in second and "GOOD" not in first else "tie"
            )
            return {
                wrapper_key: {
                    "winner": winner,
                    "first_has_unique_factual_error": "WRONG" in first,
                    "second_has_unique_factual_error": "WRONG" in second,
                    "evidence_issues": "",
                    "rationale": f"wrapped {winner}",
                }
            }

        judgement, _ = await _judge(runs, judge=FakeJudge(decide=wrapped))
        plain, _ = await _judge(runs)
        assert {q.status for q in judgement.questions} == {"judged"}
        assert [(q.winner, q.a_unique_factual_error, q.b_unique_factual_error) for q in judgement.questions] == [
            (q.winner, q.a_unique_factual_error, q.b_unique_factual_error) for q in plain.questions
        ]
        assert all(q.rationale.startswith("wrapped") for q in judgement.questions)

    async def test_a_single_key_dict_without_verdict_fields_is_still_an_error(self, runs):
        judgement, _ = await _judge(runs, judge=FakeJudge(decide=lambda text: {"result": {"foo": "bar"}}))
        assert {q.status for q in judgement.questions} == {"error"}

    async def test_an_unparsable_verdict_is_an_error_not_a_win(self, runs):
        judgement, _ = await _judge(runs, judge=FakeJudge(decide=lambda text: None))
        assert {q.status for q in judgement.questions} == {"error"}
        assert all(q.winner is None for q in judgement.questions)

    async def test_runs_on_different_question_sets_are_refused(self, tmp_path):
        a = write_run(tmp_path, "run-a")
        b = write_run(tmp_path, "run-b", question_set_hash="a-different-question-set")
        judge = FakeJudge()
        with pytest.raises(BenchmarkError, match="different question sets"):
            await judge_runs(load_run(a), load_run(b), judge, "judge-model-x", SEED)
        assert judge.prompts == []


def _fails_on(qid: str):
    """A verdict function whose reply about `qid` does not parse (a paid call that still errors,
    like a real wrapped-verdict ValidationError); other calls follow FakeJudge's rules."""

    def decide(text):
        if f"({qid})" in text:
            return {"result": {"foo": "bar"}}
        return FakeJudge().decide([SimpleNamespace(content=text)])

    return decide


class TestRejudgeErrors:
    """A re-run asks the judge again only about the questions whose judge call failed."""

    async def _with_one_error(self, runs, usage=(1000, 100)):
        judge = FakeJudge(decide=_fails_on("q-8k"), usage=usage)
        judgement = await judge_runs(*runs, judge, "judge-model-x", SEED, pricing=PRICED, progress=lambda _m: None)
        assert [q.id for q in judgement.questions if q.status == "error"] == ["q-8k"]
        return judgement

    async def _rejudge(self, runs, previous, judge=None, seed=999):
        judge = judge or FakeJudge(usage=(1000, 100))
        judgement = await judge_runs(
            *runs, judge, "judge-model-x", seed, pricing=PRICED, progress=lambda _m: None, previous=previous
        )
        return judgement, judge

    async def test_only_the_errored_question_is_asked_again_and_the_rest_are_kept(self, runs):
        previous = await self._with_one_error(runs)
        judgement, judge = await self._rejudge(runs, previous)

        assert len(judge.prompts) == 1 and "(q-8k)" in judge.prompts[0]
        assert {q.status for q in judgement.questions} == {"judged"}
        assert [q.id for q in judgement.questions] == [q.id for q in previous.questions]
        kept = [q for q in previous.questions if q.id != "q-8k"]
        assert [q for q in judgement.questions if q.id != "q-8k"] == kept
        # The recorded seed is kept, so the re-asked question's A/B order is the replayable one.
        assert judgement.seed == SEED
        assert next(q for q in judgement.questions if q.id == "q-8k").a_is_first == a_is_first(SEED, "q-8k")

    async def test_the_spend_of_the_earlier_calls_is_kept_in_the_totals(self, runs):
        previous = await self._with_one_error(runs)
        judgement, _ = await self._rejudge(runs, previous)
        assert judgement.judge_input_tokens == previous.judge_input_tokens + 1000
        assert judgement.judge_output_tokens == previous.judge_output_tokens + 100
        assert judgement.judge_cost_usd == pytest.approx(previous.judge_cost_usd + (1000 * 2 + 100 * 10) / 1e6)

    async def test_an_unknown_earlier_cost_keeps_the_total_unknown(self, runs):
        previous = await self._with_one_error(runs, usage=None)
        assert previous.judge_cost_usd is None
        judgement, _ = await self._rejudge(runs, previous)
        assert judgement.judge_cost_usd is None

    @pytest.mark.parametrize("change", ["judge", "runs"])
    async def test_a_judgement_by_another_judge_or_of_other_runs_is_not_completed(self, runs, change):
        previous = await self._with_one_error(runs)
        if change == "judge":
            previous = previous.model_copy(update={"judge_model": "another-judge"})
        else:
            previous = previous.model_copy(update={"a": previous.a.model_copy(update={"run_id": "other-run"})})
        judge = FakeJudge()
        with pytest.raises(BenchmarkError, match="Cannot complete the earlier judgement"):
            await self._rejudge(runs, previous, judge)
        assert judge.prompts == []
