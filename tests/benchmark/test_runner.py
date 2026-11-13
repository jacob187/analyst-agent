"""AC2 (run writes per-question JSON from captured spans), AC8 (no real LLMs).

Architectures are test-registered (the real `current` builder is never reached). The agents
run a real LangGraph ReAct graph on a scripted chat model, so spans are genuine
OpenInference spans.
"""

import json

import pytest

from evals.benchmark import architectures
from evals.benchmark.architectures import ARCHITECTURES, get_builder
from evals.benchmark.cli import main
from evals.benchmark.common import BenchmarkError, provider_env_var, resolve_model
from evals.benchmark.pricing import pricing_from_model
from evals.benchmark.questions import Question, question_set_hash
from evals.benchmark.runner import preflight, run_benchmark
from evals.benchmark.runs import load_run
from tests.benchmark.helpers import FakeAgent

pytestmark = pytest.mark.eval_unit

SECRET = "sk-test-secret-value-1234567890"


class _NeverRespondsAgent:
    async def stream(self, inputs):
        yield {"type": "node", "node": "router", "message": "routing"}


class _MidStreamFailureAgent(FakeAgent):
    async def stream(self, inputs):
        async for event in super().stream(inputs):
            if event["type"] == "response":
                raise RuntimeError("synthesizer blew up")
            yield event


class _HangingAgent:
    async def stream(self, inputs):
        import asyncio

        await asyncio.sleep(30)
        yield {"type": "response", "message": "too late"}


class _ThreadPoolAgent:
    """Calls its LLM from a plain executor thread, which does not copy contextvars,
    like sec_tools._tool_all_summaries: the LLM span lands in a separate trace."""

    async def stream(self, inputs):
        import asyncio

        from langchain_core.messages import AIMessage

        from tests.benchmark.helpers import _ScriptedChatModel

        usage = {"input_tokens": 40, "output_tokens": 5, "total_tokens": 45}
        llm = _ScriptedChatModel(responses=[AIMessage(content="from a thread", usage_metadata=usage)])
        reply = await asyncio.get_running_loop().run_in_executor(None, llm.invoke, "hi")
        yield {"type": "response", "message": reply.content}


async def _build(ticker: str, model_id: str):
    if ticker == "THREAD":
        return _ThreadPoolAgent()
    if ticker == "BUILD":
        raise RuntimeError(f"cannot build agent, key={SECRET}")
    if ticker == "HALF":
        return _MidStreamFailureAgent()
    if ticker == "NORESP":
        return _NeverRespondsAgent()
    if ticker == "HANG":
        return _HangingAgent()
    if ticker == "NOUSAGE":
        return FakeAgent(usage=False)
    return FakeAgent(f"Answer about {ticker}.")


def _q(qid: str, ticker: str, category: str = "technicals") -> Question:
    return Question(id=qid, ticker=ticker, category=category, question=f"Question {qid}?")


@pytest.fixture
def fake_arch(monkeypatch):
    monkeypatch.setitem(ARCHITECTURES, "fake", _build)
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)  # also exercises secret scrubbing
    return "fake"


@pytest.fixture
def model():
    return resolve_model(None)


class TestRun:
    async def test_writes_run_json_and_per_question_json_from_spans(self, fake_arch, model, tmp_path):
        questions = [_q("q1", "AAPL"), _q("q2", "NVDA", "market_news")]
        out = tmp_path / "run"
        meta = await run_benchmark(fake_arch, model, questions, out, progress=lambda _m: None)

        run = load_run(out)
        assert run.meta == meta
        assert meta.architecture == "fake"
        assert meta.model_id == model.id
        assert meta.question_set_hash == question_set_hash(questions)
        assert meta.question_ids == ["q1", "q2"]
        assert meta.tavily_enabled is True
        assert meta.pricing == pricing_from_model(model)
        assert meta.git_sha and meta.date and meta.finished_at and meta.run_id
        assert meta.notes  # the shared-cache caveat is recorded

        q1 = run.questions["q1"]
        assert (q1.status, q1.answer) == ("ok", "Answer about AAPL.")
        assert (q1.ticker, q1.category, q1.question) == ("AAPL", "technicals", "Question q1?")
        assert q1.llm_calls == 2
        assert (q1.input_tokens, q1.output_tokens, q1.total_tokens) == (250, 30, 280)
        assert q1.llm_calls_without_usage == 0
        assert [t.name for t in q1.tool_calls] == ["lookup_fact"]
        assert q1.latency_s > 0 and q1.llm_time_s > 0
        assert q1.errors == []
        # Raw file really is JSON on disk.
        assert json.loads((out / "questions" / "q2.json").read_text())["category"] == "market_news"

    async def test_one_failing_question_does_not_abort_the_run(self, fake_arch, model, tmp_path):
        questions = [
            _q("ok-1", "AAPL"),
            _q("build-fails", "BUILD"),
            _q("stream-fails", "HALF"),
            _q("no-response", "NORESP"),
            _q("ok-2", "JPM"),
        ]
        out = tmp_path / "run"
        await run_benchmark(fake_arch, model, questions, out, progress=lambda _m: None)

        run = load_run(out)
        assert set(run.questions) == {q.id for q in questions}
        assert run.questions["ok-1"].status == run.questions["ok-2"].status == "ok"
        for qid in ("build-fails", "stream-fails", "no-response"):
            assert run.questions[qid].status == "failed"
            assert run.questions[qid].answer is None
            assert run.questions[qid].errors
        assert "synthesizer blew up" in run.questions["stream-fails"].errors[0]
        # Metrics gathered before the failure are kept.
        assert run.questions["stream-fails"].llm_calls == 2
        assert "without a response event" in run.questions["no-response"].errors[0]

    async def test_a_hanging_question_times_out_and_the_run_continues(self, fake_arch, model, tmp_path):
        out = tmp_path / "run"
        await run_benchmark(
            fake_arch, model, [_q("hang", "HANG"), _q("after", "AAPL")], out,
            timeout_s=0.2, progress=lambda _m: None,
        )
        run = load_run(out)
        assert run.questions["hang"].status == "failed"
        assert "TimeoutError" in run.questions["hang"].errors[0]
        assert run.questions["after"].status == "ok"

    async def test_api_keys_never_reach_the_output_files(self, fake_arch, model, tmp_path, monkeypatch):
        monkeypatch.setenv(provider_env_var(model), SECRET)
        out = tmp_path / "run"
        await run_benchmark(fake_arch, model, [_q("build-fails", "BUILD")], out, progress=lambda _m: None)

        written = "".join(p.read_text() for p in out.rglob("*.json"))
        assert "cannot build agent" in written
        assert SECRET not in written
        assert "[redacted]" in written

    async def test_missing_token_usage_is_flagged_not_zeroed(self, fake_arch, model, tmp_path):
        out = tmp_path / "run"
        await run_benchmark(fake_arch, model, [_q("nousage", "NOUSAGE")], out, progress=lambda _m: None)
        result = load_run(out).questions["nousage"]
        assert result.llm_calls == 2
        assert result.llm_calls_without_usage == 2
        assert result.total_tokens == 0

    async def test_llm_calls_on_threads_outside_the_question_trace_are_still_counted(self, fake_arch, model, tmp_path):
        out = tmp_path / "run"
        await run_benchmark(fake_arch, model, [_q("threaded", "THREAD")], out, progress=lambda _m: None)
        result = load_run(out).questions["threaded"]
        assert result.status == "ok"
        assert result.llm_calls == 1
        assert (result.input_tokens, result.output_tokens) == (40, 5)
        assert result.spans_outside_root_trace == 1

    async def test_only_runs_the_selected_ids(self, fake_arch, model, tmp_path):
        questions = [_q("q1", "AAPL"), _q("q2", "NVDA"), _q("q3", "JPM")]
        out = tmp_path / "run"
        meta = await run_benchmark(fake_arch, model, questions, out, only=["q3", "q1"], progress=lambda _m: None)
        assert meta.question_ids == ["q1", "q3"]
        assert set(load_run(out).questions) == {"q1", "q3"}
        # The hash covers the whole file, not the subset.
        assert meta.question_set_hash == question_set_hash(questions)

    async def test_unknown_only_id_is_rejected_before_anything_runs(self, fake_arch, model, tmp_path):
        out = tmp_path / "run"
        with pytest.raises(BenchmarkError, match="Unknown question id.*nope"):
            await run_benchmark(fake_arch, model, [_q("q1", "AAPL")], out, only=["nope"])
        assert not out.exists()

    async def test_refuses_to_overwrite_an_existing_run(self, fake_arch, model, tmp_path):
        out = tmp_path / "run"
        await run_benchmark(fake_arch, model, [_q("q1", "AAPL")], out, progress=lambda _m: None)
        with pytest.raises(BenchmarkError, match="already holds a run"):
            await run_benchmark(fake_arch, model, [_q("q1", "AAPL")], out, progress=lambda _m: None)


class TestPreflight:
    def test_unknown_architecture(self):
        with pytest.raises(BenchmarkError, match="Unknown architecture 'nope'.*current"):
            preflight("nope", None)

    def test_unknown_model(self, monkeypatch):
        monkeypatch.setitem(ARCHITECTURES, "fake", _build)
        with pytest.raises(BenchmarkError, match="Unknown model 'no-such-model'"):
            preflight("fake", "no-such-model")

    def test_missing_provider_key_is_reported_by_env_var_name(self, monkeypatch):
        monkeypatch.setitem(ARCHITECTURES, "fake", _build)
        env_var = provider_env_var(resolve_model(None))
        monkeypatch.delenv(env_var, raising=False)
        with pytest.raises(BenchmarkError, match=f"{env_var} is not set"):
            preflight("fake", None)

    def test_with_key_returns_builder_and_default_model(self, monkeypatch):
        monkeypatch.setitem(ARCHITECTURES, "fake", _build)
        monkeypatch.setenv(provider_env_var(resolve_model(None)), SECRET)
        builder, model = preflight("fake", None)
        assert builder is _build
        assert model == resolve_model(None)

    def test_current_is_registered(self):
        assert get_builder("current") is architectures.build_current


class TestCliErrors:
    def test_unknown_model_exits_2_with_a_clear_message(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("evals.benchmark.cli.prepare_environment", lambda: None)  # no .env reload
        code = main(["run", "--graph", "current", "--model", "no-such-model", "--out", str(tmp_path / "r")])
        assert code == 2
        assert "Unknown model 'no-such-model'" in capsys.readouterr().err
        assert not (tmp_path / "r").exists()

    def test_missing_key_exits_2_before_running_anything(self, tmp_path, capsys, monkeypatch):
        # prepare_environment() would reload the project .env and restore the key
        # (and a real `current` run would follow), so it is neutralised here.
        monkeypatch.setattr("evals.benchmark.cli.prepare_environment", lambda: None)
        monkeypatch.delenv(provider_env_var(resolve_model(None)), raising=False)
        code = main(["run", "--graph", "current", "--out", str(tmp_path / "r")])
        assert code == 2
        assert "is not set" in capsys.readouterr().err
        assert not (tmp_path / "r").exists()
