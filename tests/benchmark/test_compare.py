"""AC1 (model list from flags or keys), AC2 (judged against a reference), AC3 (summary + PDF files),
AC5 (cost estimate and confirmation), resumability and the guards of `compare-models`.

Architectures are test-registered fakes (a real LangGraph ReAct agent on a scripted chat model) and
the judge is the `FakeJudge`; no test can reach a real LLM or the network. The `.env` reload in
`prepare_environment` is neutralised the way `TestCliErrors` does it.
"""

import asyncio
import io
import json
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from agents.model_registry import get_all_models
from evals.benchmark.architectures import ARCHITECTURES
from evals.benchmark.cli import main
from evals.benchmark.common import BenchmarkError, provider_env_var
from evals.benchmark.compare import (
    ASSUMED_INPUT_TOKENS_PER_QUESTION,
    ASSUMED_JUDGE_INPUT_TOKENS_PER_PAIR,
    ASSUMED_JUDGE_OUTPUT_TOKENS_PER_PAIR,
    ASSUMED_OUTPUT_TOKENS_PER_QUESTION,
    RUNS_DIR,
    confirm,
    default_out_dir,
    select_models,
)
from evals.benchmark.judge import judge_runs
from evals.benchmark.runs import load_judgement, load_run, write_model
from tests.benchmark.helpers import FakeAgent, FakeJudge, pdf_text_blob, write_run
from tests.conftest import clear_operator_keys, flagged_free_tier_model, set_operator_key

pytestmark = pytest.mark.eval_unit

# The `env` fixture drops ANON_FREE_MODEL_ID, so the command sees the flagged model.
FREE = flagged_free_tier_model()
ALL = get_all_models()
SAME_PROVIDER = [m for m in ALL if m.provider == FREE.provider and m.id != FREE.id]
PAID = next(m for m in SAME_PROVIDER if m.input_price_per_mtok is not None)
PAID2 = next(m for m in SAME_PROVIDER if m.id != PAID.id)
UNPRICED_MODEL = next(m for m in ALL if m.input_price_per_mtok is None and m.provider != FREE.provider)
OTHER_PROVIDER = [m for m in ALL if m.provider == UNPRICED_MODEL.provider]

QUESTIONS = [
    ("q1", "AAPL", "filing_lookup"),
    ("q2", "AAPL", "filing_lookup"),
    ("q3", "NVDA", "multi_step"),
    ("q4", "JPM", "material_events"),
    ("q5", "AAPL", "technicals"),
    ("q6", "NVDA", "market_news"),
    ("q7", "MSFT", "multi_step"),
]
N = len(QUESTIONS)
# FakeAgent: two LLM calls, 100+150 input and 10+20 output tokens per question.
TOKENS_IN, TOKENS_OUT = 250, 30

STYLE: dict[str, str] = {}
BUILDS: list[str] = []
JUDGE_LLMS: list[object] = []
# {"tokens": (in, out)} makes the fake judge report that usage per call (absent: no usage reported).
JUDGE_USAGE: dict[str, tuple[int, int]] = {}
# {"fn": decide} replaces the fake judge's verdict rule (see FakeJudge).
JUDGE_DECIDE: dict[str, object] = {}


FAILING: set[str] = set()  # models whose agent cannot be built (like an exhausted provider quota)


async def _build(ticker: str, model_id: str):
    BUILDS.append(model_id)
    if model_id in FAILING:
        raise RuntimeError("RateLimitError: 429 no credits remaining")
    return FakeAgent(f"{STYLE.get(model_id, 'plain')} answer about {ticker}")


def _cost(model, tokens_in: float, tokens_out: float, n: int = N) -> str:
    usd = n * (tokens_in * model.input_price_per_mtok + tokens_out * model.output_price_per_mtok) / 1e6
    return f"${usd:.4f}"


class _UnreadableStdin:
    def readline(self, *_):
        raise AssertionError("the command must not ask for confirmation here")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
    clear_operator_keys(monkeypatch)
    monkeypatch.setattr("evals.benchmark.cli.prepare_environment", lambda: None)  # no .env reload
    monkeypatch.setitem(ARCHITECTURES, "fake", _build)
    monkeypatch.setattr(sys, "stdin", _UnreadableStdin())
    STYLE.clear()
    FAILING.clear()
    BUILDS.clear()
    JUDGE_LLMS.clear()
    JUDGE_USAGE.clear()
    JUDGE_DECIDE.clear()

    def judge_factory(model):
        judge = FakeJudge(decide=JUDGE_DECIDE.get("fn"), usage=JUDGE_USAGE.get("tokens"))
        JUDGE_LLMS.append(judge)
        return judge

    monkeypatch.setattr("evals.benchmark.judge.build_judge_llm", judge_factory)


def give_keys(monkeypatch, *models):
    for model in models:
        set_operator_key(monkeypatch, model)


@pytest.fixture
def qfile(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(
        yaml.safe_dump([{"id": i, "ticker": t, "category": c, "question": f"Question {i}?"} for i, t, c in QUESTIONS])
    )
    return path


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


def compare(qfile, out, *extra):
    return main(["compare-models", "--graph", "fake", "--questions", str(qfile), "--out", str(out), *extra])


class TestSelectModels:
    def test_defaults_to_every_registry_model_with_a_key(self, monkeypatch):
        give_keys(monkeypatch, FREE)
        reference, models = select_models(None, None)
        assert reference.id == FREE.id
        assert [m.id for m in models] == [FREE.id, *[m.id for m in SAME_PROVIDER]]
        assert all(m.provider == FREE.provider for m in models)

    def test_models_of_every_provider_with_a_key_are_included(self, monkeypatch):
        give_keys(monkeypatch, FREE, UNPRICED_MODEL)
        _, models = select_models(None, None)
        assert {m.id for m in models} == {m.id for m in ALL if m.provider in {FREE.provider, UNPRICED_MODEL.provider}}

    def test_explicit_list_is_used_and_the_reference_is_added_first(self, monkeypatch):
        give_keys(monkeypatch, FREE)
        reference, models = select_models([PAID.id, PAID2.id, PAID.id], None)
        assert reference.id == FREE.id
        assert [m.id for m in models] == [FREE.id, PAID.id, PAID2.id]

    def test_reference_flag_overrides_the_free_tier_default(self, monkeypatch):
        reference, models = select_models([PAID.id], PAID2.id)
        assert reference.id == PAID2.id
        assert [m.id for m in models] == [PAID2.id, PAID.id]

    def test_reference_already_in_the_list_is_not_duplicated(self):
        _, models = select_models([PAID.id, FREE.id], None)
        assert [m.id for m in models] == [FREE.id, PAID.id]

    def test_no_keys_and_no_list_is_an_error(self):
        with pytest.raises(BenchmarkError, match="No registry model has its provider API key"):
            select_models(None, None)

    def test_unknown_model_id_is_an_error(self):
        with pytest.raises(BenchmarkError, match="Unknown model 'no-such-model'"):
            select_models(["no-such-model"], None)

    def test_out_dir_default_is_stable_per_graph(self, tmp_path):
        assert default_out_dir(tmp_path, "g") == tmp_path / "compare-g"


class TestFullRun:
    def test_runs_each_model_judges_it_against_the_reference_and_writes_everything(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        STYLE[PAID.id] = "GOOD"
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0

        for model in (FREE, PAID):
            run = load_run(out / "runs" / model.id)
            assert run.meta.model_id == model.id and run.meta.finished_at
            assert set(run.questions) == {q[0] for q in QUESTIONS}

        judgement_dir = out / "judgements" / f"{PAID.id}-vs-{FREE.id}"
        judgement = load_judgement(judgement_dir)
        assert judgement.a.model_id == PAID.id and judgement.b.model_id == FREE.id
        assert judgement.judge_model == FREE.id
        assert sum(q.status == "judged" and q.winner == "a" for q in judgement.questions) == N

        summary = json.loads((out / "summary.json").read_text())
        assert summary["reference"] == FREE.id and summary["graph"] == "fake"
        assert list(summary["models"]) == [FREE.id, PAID.id]
        paid = summary["models"][PAID.id]
        assert paid["vs_reference"]["wins"] == N and paid["vs_reference"]["unique_factual_errors"] == 0
        assert paid["means"]["llm_calls"] == 2
        assert paid["means"]["input_tokens"] == TOKENS_IN and paid["means"]["output_tokens"] == TOKENS_OUT
        assert paid["means"]["cost_usd"] == pytest.approx(
            (TOKENS_IN * PAID.input_price_per_mtok + TOKENS_OUT * PAID.output_price_per_mtok) / 1e6
        )
        assert paid["prices_as_of"] == PAID.prices_as_of
        assert set(paid["by_category"]) == {q[2] for q in QUESTIONS}
        assert set(summary["recommendations"]) == {"chat_default", "free_trial", "filing_analysis"}

        pdf = (out / "report.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-")
        text = pdf_text_blob(pdf)
        assert all(q[0] in text for q in QUESTIONS)
        assert FREE.id in text and PAID.id in text

        printed = capsys.readouterr().out
        assert "Recommendations:" in printed and "Rules:" in printed and "free_trial" in printed

    def test_the_judge_only_sees_answers_never_model_ids(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        (judge,) = JUDGE_LLMS
        assert len(judge.prompts) == N
        assert all(FREE.id not in p and PAID.id not in p for p in judge.prompts)

    def test_defaults_to_every_model_the_operator_has_a_key_for(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--only", "q1,q2", "--yes") == 0
        ran = {p.name for p in (out / "runs").iterdir()}
        assert ran == {FREE.id, *[m.id for m in SAME_PROVIDER]}
        judged = {p.name for p in (out / "judgements").iterdir()}
        assert judged == {f"{m.id}-vs-{FREE.id}" for m in SAME_PROVIDER}
        assert json.loads((out / "summary.json").read_text())["question_ids"] == ["q1", "q2"]

    def test_reference_flag_judges_against_that_model_and_runs_it(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--reference", PAID2.id, "--yes") == 0
        assert {p.name for p in (out / "runs").iterdir()} == {PAID.id, PAID2.id}  # free tier not run
        assert (out / "judgements" / f"{PAID.id}-vs-{PAID2.id}" / "judgement.json").exists()
        assert json.loads((out / "summary.json").read_text())["reference"] == PAID2.id

    def test_compare_report_rerenders_the_pdf_from_a_finished_directory(self, monkeypatch, qfile, out, tmp_path):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        (out / "report.pdf").unlink()
        assert main(["compare-report", "--dir", str(out)]) == 0
        assert (out / "report.pdf").read_bytes().startswith(b"%PDF-")
        other = tmp_path / "x" / "again.pdf"
        assert main(["compare-report", "--dir", str(out), "--out", str(other)]) == 0
        assert other.exists()

    def test_compare_report_on_a_missing_directory_exits_2(self, tmp_path, capsys):
        assert main(["compare-report", "--dir", str(tmp_path / "nope")]) == 2
        assert "summary.json" in capsys.readouterr().err

    def test_appendix_holds_the_answers_of_questions_not_featured_in_the_main_section(
        self, monkeypatch, qfile, out
    ):
        give_keys(monkeypatch, FREE)
        STYLE[PAID.id] = "GOOD"
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        text = pdf_text_blob((out / "report.pdf").read_bytes())
        # q2 (second filing_lookup) and q7 (second multi_step) are only in the appendix.
        assert "Appendix B" in text
        assert "GOOD answer about MSFT" in text  # q7


class TestCostConfirmation:
    def test_dry_run_prints_the_estimate_per_model_and_overall_and_runs_nothing(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--dry-run") == 0
        printed = capsys.readouterr().out
        assert f"run {FREE.id}: " in printed and f"run {PAID.id}: " in printed
        assert _cost(PAID, ASSUMED_INPUT_TOKENS_PER_QUESTION, ASSUMED_OUTPUT_TOKENS_PER_QUESTION) in printed
        assert _cost(FREE, ASSUMED_INPUT_TOKENS_PER_QUESTION, ASSUMED_OUTPUT_TOKENS_PER_QUESTION) in printed
        assert "judge x1" in printed
        assert "TOTAL: $" in printed
        assert f"prices as of {PAID.prices_as_of}" in printed
        assert "ROUGH GUESS" in printed  # no earlier run to measure
        assert "Dry run" in printed
        assert not (out / RUNS_DIR).exists() and not out.exists()
        assert BUILDS == [] and JUDGE_LLMS == []

    def test_estimate_uses_token_counts_from_an_earlier_run(self, monkeypatch, qfile, out, tmp_path, capsys):
        give_keys(monkeypatch, FREE)
        prior = write_run(tmp_path, "prior")  # 1000 input / 200 output tokens per question
        assert compare(qfile, out, "--models", PAID.id, "--estimate-from", str(prior), "--dry-run") == 0
        printed = capsys.readouterr().out
        assert _cost(PAID, 1000, 200) in printed
        assert f"mean tokens per question from run {prior}" in printed
        assert "ROUGH GUESS" not in printed

    def test_estimate_defaults_to_the_reference_run_when_one_exists(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id) == 0  # free tier only: no prompt
        capsys.readouterr()
        assert compare(qfile, out, "--models", PAID.id, "--dry-run") == 0
        printed = capsys.readouterr().out
        assert _cost(PAID, TOKENS_IN, TOKENS_OUT) in printed
        assert "the reference run" in printed
        assert f"run {FREE.id}: already complete, skipped" in printed

    def test_an_unpriced_model_prints_unknown(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE, UNPRICED_MODEL)
        assert compare(qfile, out, "--models", UNPRICED_MODEL.id, "--dry-run") == 0
        printed = capsys.readouterr().out
        assert f"run {UNPRICED_MODEL.id}: unknown (price missing)" in printed
        assert "the real total is unknown" in printed

    def test_refuses_without_yes_when_stdin_is_not_confirmed(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF: nobody to confirm
        assert compare(qfile, out, "--models", PAID.id) == 2
        assert "Not confirmed" in capsys.readouterr().err
        assert not out.exists()
        assert BUILDS == [] and JUDGE_LLMS == []

    def test_an_unpriced_model_still_requires_confirmation(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE, UNPRICED_MODEL)
        monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
        assert compare(qfile, out, "--models", UNPRICED_MODEL.id) == 2
        assert not out.exists() and BUILDS == []

    def test_a_typed_yes_confirms(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        monkeypatch.setattr(sys, "stdin", io.StringIO("yes\n"))
        assert compare(qfile, out, "--models", PAID.id) == 0
        assert (out / "summary.json").exists()

    def test_yes_skips_the_prompt(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0  # stdin would fail the test if read

    def test_a_plan_with_only_the_free_tier_model_does_not_prompt(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id) == 0  # stdin would fail the test if read
        assert (out / "summary.json").exists()

    @pytest.mark.parametrize("stdin_error", [EOFError, OSError])
    def test_confirm_treats_unreadable_stdin_as_no(self, monkeypatch, stdin_error):
        class Broken:
            def readline(self, *_):
                raise stdin_error()

        monkeypatch.setattr(sys, "stdin", Broken())
        assert confirm("? ") is False

    def test_confirm_only_accepts_yes(self, monkeypatch):
        for answer, expected in [("y\n", True), ("YES\n", True), ("\n", False), ("yep\n", False), ("no\n", False)]:
            monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
            assert confirm("? ") is expected, answer


class TestResume:
    def test_a_second_invocation_skips_complete_runs_and_judgements_and_says_so(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        run_ids = {m: load_run(out / "runs" / m).meta.run_id for m in (FREE.id, PAID.id)}
        judged_at = load_judgement(out / "judgements" / f"{PAID.id}-vs-{FREE.id}").date
        BUILDS.clear()
        JUDGE_LLMS.clear()
        capsys.readouterr()

        assert compare(qfile, out, "--models", PAID.id) == 0  # nothing pending, so no prompt either
        printed = capsys.readouterr().out
        assert f"{FREE.id}: complete run found" in printed and f"{PAID.id}: complete run found" in printed
        assert "judgement found" in printed
        assert BUILDS == [] and JUDGE_LLMS == []
        assert {m: load_run(out / "runs" / m).meta.run_id for m in (FREE.id, PAID.id)} == run_ids
        assert load_judgement(out / "judgements" / f"{PAID.id}-vs-{FREE.id}").date == judged_at
        assert (out / "summary.json").exists()

    def test_only_the_new_model_runs_and_is_judged_when_the_list_grows(self, monkeypatch, qfile, out):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        BUILDS.clear()
        assert compare(qfile, out, "--models", f"{PAID.id},{PAID2.id}", "--yes") == 0
        assert set(BUILDS) == {PAID2.id}
        assert (out / "judgements" / f"{PAID2.id}-vs-{FREE.id}").exists()
        assert list(json.loads((out / "summary.json").read_text())["models"]) == [FREE.id, PAID.id, PAID2.id]

    def test_an_incomplete_run_is_refused_with_the_directory_to_delete(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        (out / "runs" / PAID.id / "questions" / "q3.json").unlink()
        BUILDS.clear()
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 2
        err = capsys.readouterr().err
        assert "incomplete run" in err and str(out / "runs" / PAID.id) in err
        assert BUILDS == []

    def test_a_run_that_never_finished_is_incomplete_too(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id) == 0
        run = load_run(out / "runs" / FREE.id)
        run.meta.finished_at = None
        write_model(out / "runs" / FREE.id / "run.json", run.meta)
        assert compare(qfile, out, "--models", FREE.id) == 2
        assert "incomplete run" in capsys.readouterr().err

    def test_a_directory_without_a_run_file_is_not_overwritten(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        stray = out / "runs" / FREE.id
        stray.mkdir(parents=True)
        (stray / "notes.txt").write_text("keep me")
        assert compare(qfile, out, "--models", FREE.id) == 2
        assert "holds no run.json" in capsys.readouterr().err
        assert (stray / "notes.txt").read_text() == "keep me"

    def test_a_complete_run_of_different_work_is_refused_not_skipped(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id, "--only", "q1,q2") == 0
        run_json = (out / "runs" / FREE.id / "run.json").read_text()
        BUILDS.clear()
        assert compare(qfile, out, "--models", FREE.id) == 2  # now all questions
        err = capsys.readouterr().err
        assert "differs from this invocation" in err and "question_ids" in err
        assert BUILDS == []
        assert (out / "runs" / FREE.id / "run.json").read_text() == run_json

    def test_a_changed_question_set_is_refused(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id) == 0
        qfile.write_text(yaml.safe_dump([{"id": i, "ticker": t, "category": c, "question": f"Changed {i}?"} for i, t, c in QUESTIONS]))
        assert compare(qfile, out, "--models", FREE.id) == 2
        assert "question_set_hash" in capsys.readouterr().err

    def test_a_run_of_another_architecture_is_refused(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id) == 0
        monkeypatch.setitem(ARCHITECTURES, "other-fake", _build)
        code = main(["compare-models", "--graph", "other-fake", "--questions", str(qfile), "--out", str(out), "--models", FREE.id])
        assert code == 2
        assert "architecture" in capsys.readouterr().err

    def test_a_judgement_of_other_runs_is_refused_before_anything_runs(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        import shutil

        shutil.rmtree(out / "runs" / PAID.id)  # its judgement now refers to a run that no longer exists
        BUILDS.clear()
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 2
        assert "holds a judgement but a run it needs is missing" in capsys.readouterr().err
        assert BUILDS == []

    def test_resuming_with_another_judge_model_is_refused_before_anything_runs(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0  # judged by the free-tier default
        BUILDS.clear()
        JUDGE_LLMS.clear()
        capsys.readouterr()

        assert compare(qfile, out, "--models", f"{PAID.id},{PAID2.id}", "--judge-model", PAID2.id, "--yes") == 2
        err = capsys.readouterr().err
        assert f"judged by {FREE.id}" in err and PAID2.id in err and "--allow-mixed-judges" in err
        assert BUILDS == [] and JUDGE_LLMS == []

    def test_mixed_judges_when_allowed_are_named_per_model_and_warned_about(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        capsys.readouterr()
        assert compare(
            qfile, out, "--models", f"{PAID.id},{PAID2.id}", "--judge-model", PAID2.id, "--allow-mixed-judges", "--yes"
        ) == 0

        summary = json.loads((out / "summary.json").read_text())
        assert summary["judge_model"] is None
        assert summary["judge_models"] == {PAID.id: FREE.id, PAID2.id: PAID2.id}
        assert any("different judge models" in w for w in summary["warnings"])
        printed = capsys.readouterr().out
        assert f"judges: {PAID.id} by {FREE.id}, {PAID2.id} by {PAID2.id}" in printed
        assert f"{PAID.id} by {FREE.id}, {PAID2.id} by {PAID2.id}" in pdf_text_blob((out / "report.pdf").read_bytes())

    def test_a_rerun_judges_only_the_errored_questions_again(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        plain = FakeJudge()

        def flaky(text):
            if "Question q3?" in text:
                return {"result": "not a verdict"}  # a paid call that fails to parse
            return plain.decide([SimpleNamespace(content=text)])

        JUDGE_DECIDE["fn"] = flaky
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        judgement_dir = out / "judgements" / f"{PAID.id}-vs-{FREE.id}"
        before = {q.id: q for q in load_judgement(judgement_dir).questions}
        assert before["q3"].status == "error"
        assert "1 judge error(s)" in capsys.readouterr().out

        JUDGE_DECIDE.clear()
        JUDGE_LLMS.clear()
        assert compare(qfile, out, "--models", PAID.id, "--dry-run") == 0
        assert "1 errored question(s) to re-judge" in capsys.readouterr().out
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0

        (judge,) = JUDGE_LLMS
        assert len(judge.prompts) == 1 and "Question q3?" in judge.prompts[0]
        after = {q.id: q for q in load_judgement(judgement_dir).questions}
        assert after["q3"].status == "judged"
        assert all(after[qid] == before[qid] for qid in before if qid != "q3")
        assert json.loads((out / "summary.json").read_text())["models"][PAID.id]["vs_reference"]["judge_errors"] == 0

    def test_a_judgement_with_other_run_ids_is_refused(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        judgement_dir = out / "judgements" / f"{PAID.id}-vs-{FREE.id}"
        judgement = load_judgement(judgement_dir)
        judgement.a.run_id = "some-other-run"
        write_model(judgement_dir / "judgement.json", judgement)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 2
        assert "judges different runs" in capsys.readouterr().err


class TestGuards:
    def test_unknown_model_id_exits_2_and_writes_nothing(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", "no-such-model", "--yes") == 2
        assert "Unknown model 'no-such-model'" in capsys.readouterr().err
        assert not out.exists()

    def test_no_keys_and_no_models_flag_exits_2(self, qfile, out, capsys):
        assert compare(qfile, out, "--yes") == 2
        assert "No registry model has its provider API key" in capsys.readouterr().err
        assert not out.exists()

    def test_an_explicit_model_without_a_key_names_the_env_var(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)  # the free-tier key only
        assert compare(qfile, out, "--models", UNPRICED_MODEL.id, "--yes") == 2
        assert f"{provider_env_var(UNPRICED_MODEL)} is not set" in capsys.readouterr().err
        assert not out.exists()

    def test_a_reference_without_a_key_exits_2_before_anything_runs(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, UNPRICED_MODEL)  # a key for the models, none for the free-tier reference
        assert compare(qfile, out, "--models", UNPRICED_MODEL.id, "--yes") == 2
        assert f"{provider_env_var(FREE)} is not set" in capsys.readouterr().err
        assert not out.exists() and BUILDS == []

    def test_a_missing_judge_key_is_caught_before_spending(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, OTHER_PROVIDER[0])
        ids = ",".join(m.id for m in OTHER_PROVIDER[:2])
        code = compare(qfile, out, "--models", ids, "--reference", OTHER_PROVIDER[0].id, "--yes")
        assert code == 2
        assert f"{provider_env_var(FREE)} is not set" in capsys.readouterr().err
        assert not out.exists() and BUILDS == []

    def test_unknown_question_id_exits_2(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id, "--only", "nope") == 2
        assert "Unknown question id(s) in --only: nope" in capsys.readouterr().err
        assert not out.exists()

    def test_unknown_architecture_exits_2(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        code = main(["compare-models", "--graph", "nope", "--questions", str(qfile), "--out", str(out), "--models", FREE.id])
        assert code == 2
        assert "Unknown architecture 'nope'" in capsys.readouterr().err

    @pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "-inf"])
    def test_cost_multiple_must_be_finite_and_positive(self, monkeypatch, qfile, out, capsys, bad):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", FREE.id, f"--chat-cost-multiple={bad}") == 2
        assert "--chat-cost-multiple" in capsys.readouterr().err
        assert not out.exists() and BUILDS == []


JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT = 2000, 50


def _judge_line(printed: str) -> str:
    return next(line.strip() for line in printed.splitlines() if line.strip().startswith("judge x"))


class TestJudgeCost:
    """AC2/AC3: judge cost lands in summary.json and the PDF, and the estimate uses measured judge tokens."""

    def test_the_judges_measured_cost_is_in_the_judgement_summary_pdf_and_total(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        JUDGE_USAGE["tokens"] = (JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0

        # The judge model is the free-tier model, which has prices: cost = measured tokens x its prices.
        expected = N * (JUDGE_TOKENS_IN * FREE.input_price_per_mtok + JUDGE_TOKENS_OUT * FREE.output_price_per_mtok) / 1e6
        judgement = load_judgement(out / "judgements" / f"{PAID.id}-vs-{FREE.id}")
        assert judgement.judge_input_tokens == N * JUDGE_TOKENS_IN
        assert judgement.judge_output_tokens == N * JUDGE_TOKENS_OUT
        assert judgement.judge_cost_usd == pytest.approx(expected)
        assert judgement.judge_prices_as_of == FREE.prices_as_of
        assert all(q.judge_input_tokens == JUDGE_TOKENS_IN for q in judgement.questions)

        summary = json.loads((out / "summary.json").read_text())
        run_cost = sum(m["totals"]["cost_usd"] for m in summary["models"].values())
        # summary.json stores USD rounded to 6 decimals; the exact value depends on registry prices.
        usd = {"abs": 1e-6}
        assert summary["costs"]["judge_usd"] == pytest.approx(expected, **usd)
        assert summary["costs"]["runs_usd"] == pytest.approx(run_cost, **usd)
        assert summary["costs"]["total_usd"] == pytest.approx(run_cost + expected, **usd)
        assert summary["models"][PAID.id]["judge"]["cost_usd"] == pytest.approx(expected, **usd)

        text = pdf_text_blob((out / "report.pdf").read_bytes())
        assert f"Judge cost: ${expected:.4f}." in text
        assert f"Total (runs + judging): ${run_cost + expected:.4f}." in text
        assert f"judging ${expected:.4f} = total ${run_cost + expected:.4f}" in capsys.readouterr().out

    def test_an_unpriced_judge_model_is_unknown_everywhere_never_zero(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE, UNPRICED_MODEL)
        JUDGE_USAGE["tokens"] = (JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT)
        assert compare(qfile, out, "--models", PAID.id, "--judge-model", UNPRICED_MODEL.id, "--yes") == 0

        judgement = load_judgement(out / "judgements" / f"{PAID.id}-vs-{FREE.id}")
        assert judgement.judge_input_tokens == N * JUDGE_TOKENS_IN and judgement.judge_cost_usd is None
        summary = json.loads((out / "summary.json").read_text())
        assert summary["costs"]["judge_usd"] is None and summary["costs"]["total_usd"] is None
        assert "Judge cost: unknown. Total (runs + judging): unknown." in pdf_text_blob((out / "report.pdf").read_bytes())
        assert "judging unknown = total unknown" in capsys.readouterr().out

    def test_the_estimate_uses_measured_judge_tokens_from_a_judgement_already_in_out(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        JUDGE_USAGE["tokens"] = (JUDGE_TOKENS_IN, JUDGE_TOKENS_OUT)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        capsys.readouterr()

        assert compare(qfile, out, "--models", f"{PAID.id},{PAID2.id}", "--dry-run") == 0
        line = _judge_line(capsys.readouterr().out)
        expected = N * (JUDGE_TOKENS_IN * FREE.input_price_per_mtok + JUDGE_TOKENS_OUT * FREE.output_price_per_mtok) / 1e6
        assert line.startswith("judge x1 (measured: mean 2,000 in / 50 out tokens per question over 7 judged question(s)")
        assert f"judge {FREE.id}" in line
        assert f"${expected:.4f}" in line
        assert "assumed" not in line

    def test_the_estimate_falls_back_to_the_labelled_assumption_without_a_measurement(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--dry-run") == 0  # nothing in out yet
        printed = capsys.readouterr().out
        line = _judge_line(printed)
        assumed = N * (
            ASSUMED_JUDGE_INPUT_TOKENS_PER_PAIR * FREE.input_price_per_mtok
            + ASSUMED_JUDGE_OUTPUT_TOKENS_PER_PAIR * FREE.output_price_per_mtok
        ) / 1e6
        assert f"assumed {ASSUMED_JUDGE_INPUT_TOKENS_PER_PAIR:,} in / {ASSUMED_JUDGE_OUTPUT_TOKENS_PER_PAIR:,} out" in line
        assert f"${assumed:.4f}" in line
        assert "the judge token counts are assumed, not measured" in printed

    def test_an_earlier_judgement_without_recorded_usage_does_not_count_as_a_measurement(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0  # the fake judge reports no usage
        capsys.readouterr()
        assert compare(qfile, out, "--models", f"{PAID.id},{PAID2.id}", "--dry-run") == 0
        line = _judge_line(capsys.readouterr().out)
        assert "assumed" in line and "measured" not in line

    def test_estimate_judge_from_points_at_a_judgement_outside_out(self, monkeypatch, qfile, out, tmp_path, capsys):
        give_keys(monkeypatch, FREE)
        a = load_run(write_run(tmp_path, "run-a"))
        b = load_run(write_run(tmp_path, "run-b"))
        prior = tmp_path / "prior-judgement"
        write_model(
            prior / "judgement.json",
            asyncio.run(judge_runs(a, b, FakeJudge(usage=(10_000, 200)), "prior-judge", 0, progress=lambda _m: None)),
        )

        assert compare(qfile, out, "--models", PAID.id, "--estimate-judge-from", str(prior), "--dry-run") == 0
        line = _judge_line(capsys.readouterr().out)
        assert "measured: mean 10,000 in / 200 out tokens per question over 5 judged question(s)" in line
        assert f"from {prior}" in line and "judge prior-judge" in line

    def test_estimate_judge_from_a_judgement_without_usage_is_refused(
        self, monkeypatch, qfile, out, tmp_path, capsys
    ):
        give_keys(monkeypatch, FREE)
        a = load_run(write_run(tmp_path, "run-a"))
        b = load_run(write_run(tmp_path, "run-b"))
        prior = tmp_path / "prior-judgement"
        write_model(prior / "judgement.json", asyncio.run(judge_runs(a, b, FakeJudge(), "j", 0, progress=lambda _m: None)))
        assert compare(qfile, out, "--models", PAID.id, "--estimate-judge-from", str(prior), "--dry-run") == 2
        assert "no judged question with recorded token usage" in capsys.readouterr().err


class TestExcludedModels:
    """A model whose run failed outright is reported as excluded and warned about, never dropped silently."""

    def test_a_run_where_every_question_failed_is_excluded_not_judged_and_warned_about(
        self, monkeypatch, qfile, out, capsys
    ):
        give_keys(monkeypatch, FREE)
        FAILING.add(PAID.id)
        assert compare(qfile, out, "--models", f"{PAID.id},{PAID2.id}", "--yes") == 0

        summary = json.loads((out / "summary.json").read_text())
        assert list(summary["models"]) == [FREE.id, PAID2.id]
        (entry,) = summary["excluded_models"]
        assert entry["model_id"] == PAID.id
        assert f"all {N} questions failed" in entry["reason"] and "RateLimitError" in entry["reason"]
        assert not (out / "judgements" / f"{PAID.id}-vs-{FREE.id}").exists()  # no judge spend on it
        (judge,) = JUDGE_LLMS
        assert len(judge.prompts) == N  # PAID2 only

        printed = capsys.readouterr().out
        assert f"WARNING: {PAID.id} excluded from the comparison" in printed
        text = pdf_text_blob((out / "report.pdf").read_bytes())
        assert PAID.id in text and "Excluded models" in text

    def test_a_reference_that_answered_nothing_stops_before_any_challenger_runs(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        FAILING.add(FREE.id)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 2
        err = capsys.readouterr().err
        assert f"The reference run {FREE.id} cannot anchor a comparison" in err
        assert f"delete {out / 'runs' / FREE.id}" in err
        assert PAID.id not in BUILDS  # no challenger was paid for
        assert not (out / "runs" / PAID.id).exists()
        assert JUDGE_LLMS == []

    def test_a_failed_reference_run_on_disk_is_refused_before_anything_runs(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        FAILING.add(FREE.id)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 2
        FAILING.clear()  # e.g. the quota is back
        BUILDS.clear()
        capsys.readouterr()

        assert compare(qfile, out, "--models", PAID.id, "--yes") == 2
        err = capsys.readouterr().err
        assert f"delete {out / 'runs' / FREE.id}" in err
        assert BUILDS == [] and JUDGE_LLMS == []

    def test_an_excluded_challenger_says_which_directory_to_delete(self, monkeypatch, qfile, out, capsys):
        give_keys(monkeypatch, FREE)
        FAILING.add(PAID.id)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        assert f"delete {out / 'runs' / PAID.id}" in capsys.readouterr().out


def _rebuild(out, summary_out, *extra, pdf=None):
    args = ["compare-report", "--dir", str(out), "--summary-out", str(summary_out), *extra]
    return main([*args, "--out", str(pdf)] if pdf else args)


class TestRebuildFromDisk:
    """`compare-report` recomputes the summary from the runs and judgements on disk: no model is called."""

    def _finished(self, monkeypatch, qfile, out, *models):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", ",".join(m.id for m in models), "--yes") == 0
        JUDGE_LLMS.clear()
        BUILDS.clear()

    def test_it_recomputes_without_running_or_judging_and_leaves_the_recorded_summary_alone(
        self, monkeypatch, qfile, out, tmp_path
    ):
        self._finished(monkeypatch, qfile, out, PAID)
        recorded = (out / "summary.json").read_text()
        scratch = tmp_path / "scratch"

        assert _rebuild(out, scratch / "summary.json", pdf=scratch / "report.pdf") == 0

        assert (out / "summary.json").read_text() == recorded
        rebuilt = json.loads((scratch / "summary.json").read_text())
        assert list(rebuilt["models"]) == [FREE.id, PAID.id]
        assert rebuilt["models"][FREE.id]["reference_standing"]["pairings_used"] == 1
        assert rebuilt["models"][PAID.id]["vs_reference"]["judged"] == N
        assert (scratch / "report.pdf").read_bytes().startswith(b"%PDF-")
        assert BUILDS == [] and JUDGE_LLMS == []

    def test_a_run_on_disk_that_the_recorded_summary_left_out_comes_back(self, monkeypatch, qfile, out, tmp_path):
        self._finished(monkeypatch, qfile, out, PAID, PAID2)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0  # this summary no longer lists PAID2
        assert PAID2.id not in json.loads((out / "summary.json").read_text())["models"]

        assert _rebuild(out, tmp_path / "s.json", pdf=tmp_path / "r.pdf") == 0
        rebuilt = json.loads((tmp_path / "s.json").read_text())
        assert PAID2.id in rebuilt["models"]

    @pytest.mark.parametrize("damage", ["removed", "emptied"])
    def test_a_missing_or_empty_run_directory_is_excluded_with_a_reason_and_a_warning(
        self, monkeypatch, qfile, out, tmp_path, capsys, damage
    ):
        self._finished(monkeypatch, qfile, out, PAID, PAID2)
        import shutil

        shutil.rmtree(out / "runs" / PAID.id)
        if damage == "emptied":
            (out / "runs" / PAID.id).mkdir()
        capsys.readouterr()

        assert _rebuild(out, tmp_path / "s.json", pdf=tmp_path / "r.pdf") == 0

        rebuilt = json.loads((tmp_path / "s.json").read_text())
        assert list(rebuilt["models"]) == [FREE.id, PAID2.id]
        (entry,) = rebuilt["excluded_models"]
        assert entry["model_id"] == PAID.id and "missing, empty or unreadable" in entry["reason"]
        assert f"WARNING: {PAID.id} excluded from the comparison" in capsys.readouterr().out

    def test_a_run_that_failed_entirely_is_excluded(self, monkeypatch, qfile, out, tmp_path, capsys):
        give_keys(monkeypatch, FREE)
        FAILING.add(PAID.id)
        assert compare(qfile, out, "--models", PAID.id, "--yes") == 0
        capsys.readouterr()

        assert _rebuild(out, tmp_path / "s.json") == 0
        (entry,) = json.loads((tmp_path / "s.json").read_text())["excluded_models"]
        assert entry["model_id"] == PAID.id and "RateLimitError" in entry["reason"]
        assert "excluded from the comparison" in capsys.readouterr().out

    def test_a_judgement_of_other_runs_is_refused(self, monkeypatch, qfile, out, tmp_path, capsys):
        self._finished(monkeypatch, qfile, out, PAID)
        judgement_dir = out / "judgements" / f"{PAID.id}-vs-{FREE.id}"
        judgement = load_judgement(judgement_dir)
        judgement.a.run_id = "some-other-run"
        write_model(judgement_dir / "judgement.json", judgement)
        assert _rebuild(out, tmp_path / "s.json") == 2
        assert "judges different runs" in capsys.readouterr().err

    @pytest.mark.parametrize("copied", ["reference", "challenger"])
    def test_a_run_directory_not_named_after_its_model_is_refused(
        self, monkeypatch, qfile, out, tmp_path, capsys, copied
    ):
        self._finished(monkeypatch, qfile, out, PAID)
        model_id = FREE.id if copied == "reference" else PAID.id
        shutil.copytree(out / "runs" / model_id, out / "runs" / f"{model_id}.bak")
        assert _rebuild(out, tmp_path / "s.json") == 2
        err = capsys.readouterr().err
        assert f"{model_id}.bak" in err and f"model_id {model_id!r}" in err
        assert not (tmp_path / "s.json").exists()

    def test_compare_report_keeps_the_previous_summary_when_the_content_changes(self, monkeypatch, qfile, out):
        self._finished(monkeypatch, qfile, out, PAID)
        recorded = (out / "summary.json").read_text()

        assert main(["compare-report", "--dir", str(out)]) == 0  # same inputs, same rules: nothing to keep
        assert not (out / "summary.prev.json").exists()
        same = (out / "summary.json").read_text()

        assert main(["compare-report", "--dir", str(out), "--chat-cost-multiple", "7"]) == 0
        assert (out / "summary.prev.json").read_text() == same
        assert json.loads((out / "summary.json").read_text())["chat_cost_multiple"] == 7
        assert json.loads(recorded)["chat_cost_multiple"] != 7

    def test_the_cost_multiple_defaults_to_the_recorded_one_and_can_be_overridden(
        self, monkeypatch, qfile, out, tmp_path
    ):
        give_keys(monkeypatch, FREE)
        assert compare(qfile, out, "--models", PAID.id, "--chat-cost-multiple", "7", "--yes") == 0
        JUDGE_LLMS.clear()
        assert _rebuild(out, tmp_path / "a.json") == 0
        assert json.loads((tmp_path / "a.json").read_text())["chat_cost_multiple"] == 7
        assert _rebuild(out, tmp_path / "b.json", "--chat-cost-multiple", "2") == 0
        assert json.loads((tmp_path / "b.json").read_text())["chat_cost_multiple"] == 2
        assert _rebuild(out, tmp_path / "c.json", "--chat-cost-multiple", "0") == 2

    @pytest.mark.parametrize("bad", ["nan", "inf"])
    def test_a_non_finite_cost_multiple_override_is_refused(self, monkeypatch, qfile, out, tmp_path, capsys, bad):
        self._finished(monkeypatch, qfile, out, PAID)
        assert _rebuild(out, tmp_path / "s.json", f"--chat-cost-multiple={bad}") == 2
        assert "--chat-cost-multiple" in capsys.readouterr().err
        assert not (tmp_path / "s.json").exists()

    def test_a_recorded_non_finite_cost_multiple_is_refused(self, monkeypatch, qfile, out, tmp_path, capsys):
        self._finished(monkeypatch, qfile, out, PAID)
        recorded = json.loads((out / "summary.json").read_text())
        recorded["chat_cost_multiple"] = float("nan")
        (out / "summary.json").write_text(json.dumps(recorded))  # what an older build wrote: a bare NaN
        assert _rebuild(out, tmp_path / "s.json") == 2
        assert "chat_cost_multiple" in capsys.readouterr().err
        assert _rebuild(out, tmp_path / "s.json", "--chat-cost-multiple", "2") == 0  # an override repairs it
