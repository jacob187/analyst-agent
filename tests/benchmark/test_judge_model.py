"""AC4: an Anthropic registry model works as `judge --model`: the judge LLM builds through
`create_llm`, structured output is forced tool use (no extended thinking), and the verdict parses.

Nothing here reaches the network: either the real ChatAnthropic is only constructed (no call), or
`init_chat_model` is replaced at the LLM boundary by a fake chat model.
"""

from types import SimpleNamespace

import pytest

from agents.llm_factory import ThinkingConfig, create_llm
from agents.model_registry import get_all_models
from evals.benchmark.cli import main
from evals.benchmark.common import provider_env_var
from evals.benchmark.judge import JudgeVerdict, build_judge_llm, judge_runs
from evals.benchmark.runs import load_judgement, load_run
from tests.benchmark.helpers import FakeJudge, write_run

pytestmark = pytest.mark.eval_unit

ANTHROPIC = [m for m in get_all_models() if m.provider == "anthropic"]
KEY = "sk-ant-test-key-0123456789"


@pytest.fixture(autouse=True)
def _no_env_reload(monkeypatch):
    monkeypatch.setattr("evals.benchmark.cli.prepare_environment", lambda: None)  # no .env reload


def _forced_tool_choice(structured) -> dict | None:
    """The `tool_choice` langchain-anthropic bound onto the request for `with_structured_output`."""
    return structured.first.kwargs.get("tool_choice")


def test_the_registry_has_an_anthropic_model():
    assert ANTHROPIC


@pytest.mark.parametrize("model", ANTHROPIC, ids=lambda m: m.id)
class TestAnthropicJudgeLlm:
    def test_the_judge_is_a_non_thinking_chat_model_and_structured_output_forces_the_verdict_tool(
        self, model, monkeypatch
    ):
        monkeypatch.setenv(provider_env_var(model), KEY)
        llm = build_judge_llm(model)

        assert type(llm).__name__ == "ChatAnthropic"
        assert llm.thinking is None
        structured = llm.with_structured_output(JudgeVerdict)
        assert _forced_tool_choice(structured) == {"type": "tool", "name": "JudgeVerdict"}

    def test_control_the_same_model_with_thinking_enabled_is_a_different_request_shape(self, model, monkeypatch):
        """Why `build_judge_llm` passes ThinkingConfig(enabled=False) rather than relying on the default:
        `create_llm(..., thinking=enabled)` turns extended thinking on for these models, and forced
        tool use (what structured output needs) is not allowed together with it."""
        monkeypatch.setenv(provider_env_var(model), KEY)
        thinking_llm = create_llm(model.id, KEY, thinking=ThinkingConfig(enabled=True, level="medium"))
        assert thinking_llm.thinking is not None
        assert build_judge_llm(model).thinking is None

    async def test_create_llm_receives_the_anthropic_provider_key_and_no_thinking_and_the_verdict_parses(
        self, model, monkeypatch, tmp_path
    ):
        calls = []

        def fake_init_chat_model(spec, **kwargs):
            calls.append((spec, kwargs))
            return FakeJudge()

        monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
        monkeypatch.setenv(provider_env_var(model), KEY)

        judge = build_judge_llm(model)

        ((spec, kwargs),) = calls
        assert spec == f"anthropic:{model.id}"
        assert kwargs["anthropic_api_key"] == KEY
        assert "thinking" not in kwargs

        answers = {"q-risk": "GOOD answer", "q-multi": "plain", "q-8k": "plain", "q-tech": "plain", "q-news": "plain"}
        a = load_run(write_run(tmp_path, "run-a", answers={k: "plain" for k in answers}))
        b = load_run(write_run(tmp_path, "run-b", answers=answers))
        judgement = await judge_runs(a, b, judge, model.id, 0, progress=lambda _m: None)
        assert judgement.judge_model == model.id
        by_id = {q.id: q for q in judgement.questions}
        assert by_id["q-risk"].status == "judged" and by_id["q-risk"].winner == "b"
        assert JudgeVerdict in judge.schemas


class TestJudgeCliWithAnAnthropicModel:
    def test_judge_command_builds_the_llm_through_create_llm_and_writes_the_judgement(
        self, monkeypatch, tmp_path
    ):
        model = ANTHROPIC[0]
        calls = []
        fake = FakeJudge()

        def fake_init_chat_model(spec, **kwargs):
            calls.append((spec, kwargs))
            return fake

        monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
        monkeypatch.setenv(provider_env_var(model), KEY)
        write_run(tmp_path, "run-a", answers={"q-risk": "plain WRONG"})
        write_run(tmp_path, "run-b", answers={"q-risk": "GOOD"})

        code = main(
            ["judge", "--a", str(tmp_path / "run-a"), "--b", str(tmp_path / "run-b"),
             "--model", model.id, "--seed", "0", "--out", str(tmp_path / "judged")]
        )

        assert code == 0
        ((spec, kwargs),) = calls
        assert spec == f"anthropic:{model.id}" and "thinking" not in kwargs
        judgement = load_judgement(tmp_path / "judged")
        assert judgement.judge_model == model.id
        assert judgement.as_of_date == "2026-02-01"
        risk = next(q for q in judgement.questions if q.id == "q-risk")
        assert risk.status == "judged" and risk.winner == "b" and risk.a_unique_factual_error is True

    def _judge(self, tmp_path, model, *extra):
        return main(["judge", "--a", str(tmp_path / "run-a"), "--b", str(tmp_path / "run-b"),
                     "--model", model.id, "--seed", "0", "--out", str(tmp_path / "judged"), *extra])

    def test_an_existing_judgement_is_not_overwritten_and_nothing_is_spent(self, monkeypatch, tmp_path, capsys):
        model = ANTHROPIC[0]
        monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec, **kw: FakeJudge())
        monkeypatch.setenv(provider_env_var(model), KEY)
        write_run(tmp_path, "run-a")
        write_run(tmp_path, "run-b")
        assert self._judge(tmp_path, model) == 0
        written = (tmp_path / "judged" / "judgement.json").read_text()
        capsys.readouterr()

        def must_not_run(*_a, **_k):
            raise AssertionError("a judge LLM was built for an --out that already holds a judgement")

        monkeypatch.setattr("langchain.chat_models.init_chat_model", must_not_run)
        assert self._judge(tmp_path, model) == 2
        err = capsys.readouterr().err
        assert "already holds" in err and "--force" in err
        assert (tmp_path / "judged" / "judgement.json").read_text() == written

    def test_a_rerun_judges_only_the_errored_questions_again(self, monkeypatch, tmp_path, capsys):
        model = ANTHROPIC[0]
        monkeypatch.setenv(provider_env_var(model), KEY)
        write_run(tmp_path, "run-a")
        write_run(tmp_path, "run-b", answers={"q-risk": "GOOD"})

        def flaky(text):
            if "(q-risk)" in text:
                raise RuntimeError("provider 529 overloaded")
            return FakeJudge().decide([SimpleNamespace(content=text)])

        monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec, **kw: FakeJudge(decide=flaky))
        assert self._judge(tmp_path, model) == 0
        before = {q.id: q for q in load_judgement(tmp_path / "judged").questions}
        assert before["q-risk"].status == "error"

        again = FakeJudge()
        monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec, **kw: again)
        assert self._judge(tmp_path, model) == 0  # no --force needed to finish the same judgement

        assert len(again.prompts) == 1 and "(q-risk)" in again.prompts[0]
        after = {q.id: q for q in load_judgement(tmp_path / "judged").questions}
        assert after["q-risk"].status == "judged" and after["q-risk"].winner == "b"
        assert all(after[qid] == before[qid] for qid in before if qid != "q-risk")

    def test_force_replaces_an_existing_judgement(self, monkeypatch, tmp_path):
        model = ANTHROPIC[0]
        monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec, **kw: FakeJudge())
        monkeypatch.setenv(provider_env_var(model), KEY)
        write_run(tmp_path, "run-a")
        write_run(tmp_path, "run-b")
        assert self._judge(tmp_path, model) == 0
        first = load_judgement(tmp_path / "judged").date
        monkeypatch.setattr("evals.benchmark.judge.utc_now", lambda: "2099-01-01T00:00:00+00:00")
        assert self._judge(tmp_path, model, "--force") == 0
        assert load_judgement(tmp_path / "judged").date == "2099-01-01T00:00:00+00:00" != first

    def test_missing_anthropic_key_is_refused_before_any_llm_is_built(self, monkeypatch, tmp_path, capsys):
        model = ANTHROPIC[0]
        monkeypatch.delenv(provider_env_var(model), raising=False)

        def must_not_run(*_a, **_k):
            raise AssertionError("an LLM was built without a key")

        monkeypatch.setattr("langchain.chat_models.init_chat_model", must_not_run)
        write_run(tmp_path, "run-a")
        write_run(tmp_path, "run-b")
        code = main(["judge", "--a", str(tmp_path / "run-a"), "--b", str(tmp_path / "run-b"),
                     "--model", model.id, "--out", str(tmp_path / "judged")])
        assert code == 2
        assert "is not set" in capsys.readouterr().err
