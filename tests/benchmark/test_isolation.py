"""Per-run database isolation (a fresh SQLite DB per run) and the OTel truncation warning.

The "tool" is a fake agent that reads and writes `filing_analyses` through the real `api.db`
helpers, as the SEC tools do. The databases are real temp-file SQLite; the default path is
redirected into `tmp_path` so no test can touch `data/chats.db`.
"""

import json
import logging

import pytest
import yaml

import api.db as db_module
from agents.model_registry import get_all_models, get_free_tier_model
from api.dependencies import PROVIDER_ENV_VARS
from api.tracing import get_tracer
from evals.benchmark.architectures import ARCHITECTURES
from evals.benchmark.cli import main
from evals.benchmark.common import provider_env_var, resolve_model
from evals.benchmark.isolation import isolated_db
from evals.benchmark.questions import Question
from evals.benchmark.runner import run_benchmark
from evals.benchmark.runs import load_run
from tests.benchmark.helpers import FakeAgent, FakeJudge

pytestmark = pytest.mark.eval_unit

FILING = dict(form_type="10-K", accession_number="0000320193-24-000123", analysis_type="risk_factors")
OTEL_LOGGER = "opentelemetry.attributes"

# One entry per lookup: (ticker, was_it_a_cache_hit).
LOOKUPS: list[tuple[str, bool]] = []


class _CachingAgent(FakeAgent):
    """Looks the filing analysis up in the DB, computing and saving it on a miss."""

    def __init__(self, ticker: str):
        super().__init__(f"Answer about {ticker}.")
        self._ticker = ticker

    async def stream(self, inputs):
        cached = await db_module.get_filing_analysis(self._ticker, **{k: FILING[k] for k in FILING})
        LOOKUPS.append((self._ticker, cached is not None))
        if cached is None:
            await db_module.save_filing_analysis(self._ticker, **FILING, analysis_json=json.dumps({"risks": []}))
        async for event in super().stream(inputs):
            yield event


async def _build(ticker: str, model_id: str):
    return _CachingAgent(ticker)


class _OversizedAttributeAgent(FakeAgent):
    async def stream(self, inputs):
        with get_tracer().start_as_current_span("filing.text", attributes={"text": "x" * 20_000}):
            pass
        async for event in super().stream(inputs):
            yield event


async def _build_oversized(ticker: str, model_id: str):
    return _OversizedAttributeAgent("Answer.")


def _q(qid: str, ticker: str = "AAPL") -> Question:
    return Question(id=qid, ticker=ticker, category="filing_lookup", question=f"Question {qid}?")


@pytest.fixture(autouse=True)
def _arch(monkeypatch):
    LOOKUPS.clear()
    monkeypatch.setitem(ARCHITECTURES, "cache", _build)
    monkeypatch.setitem(ARCHITECTURES, "oversized", _build_oversized)
    for var in PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("evals.benchmark.cli.prepare_environment", lambda: None)  # no .env reload


@pytest.fixture
async def default_db(monkeypatch, tmp_path):
    """Stands in for the dev app's `data/chats.db`; the path does not exist yet."""
    path = tmp_path / "dev" / "chats.db"
    monkeypatch.setattr(db_module, "DB_PATH", path)
    monkeypatch.setattr(db_module, "_db", None)
    yield path
    await db_module.close_db()


@pytest.fixture
def model():
    return resolve_model(None)


class TestIsolatedDb:
    async def test_second_run_does_not_see_the_first_runs_or_the_dev_apps_analyses(
        self, default_db, model, tmp_path
    ):
        # The dev app's DB holds this very analysis, and its connection is still open.
        await db_module.init_db()
        await db_module.save_filing_analysis("AAPL", **FILING, analysis_json="{}")

        questions = [_q("q1"), _q("q2")]
        await run_benchmark("cache", model, questions, tmp_path / "run1", progress=lambda _m: None)
        first = list(LOOKUPS)
        LOOKUPS.clear()
        await run_benchmark("cache", model, questions, tmp_path / "run2", progress=lambda _m: None)

        # Each run starts empty (miss) and then caches within the run (hit).
        assert first == [("AAPL", False), ("AAPL", True)]
        assert LOOKUPS == [("AAPL", False), ("AAPL", True)]

    async def test_the_dev_database_is_left_exactly_as_it_was(self, default_db, model, tmp_path):
        await db_module.init_db()
        await run_benchmark("cache", model, [_q("q1")], tmp_path / "run", progress=lambda _m: None)

        # Reusing the dev app's connection afterwards still works and holds none of the run's rows.
        assert db_module.DB_PATH == default_db
        assert await db_module.get_filing_analysis("AAPL", **FILING) is None

    async def test_a_run_never_creates_the_default_database_file(self, default_db, model, tmp_path):
        assert not default_db.parent.exists()
        await run_benchmark("cache", model, [_q("q1"), _q("q2")], tmp_path / "run", progress=lambda _m: None)

        assert not default_db.exists()
        assert not default_db.parent.exists()
        assert db_module.DB_PATH == default_db  # restored
        assert db_module._db is None  # connection closed so the process can exit

    async def test_the_run_database_is_deleted_afterwards(self, default_db, model, tmp_path):
        async with isolated_db() as path:
            assert path.exists() and db_module.DB_PATH == path
        assert not path.exists()
        assert db_module.DB_PATH == default_db

    async def test_state_is_restored_when_the_body_raises(self, default_db):
        with pytest.raises(RuntimeError, match="boom"):
            async with isolated_db():
                raise RuntimeError("boom")
        assert db_module.DB_PATH == default_db
        assert db_module._db is None

    async def test_run_json_records_the_isolated_db_and_the_note_says_so(self, default_db, model, tmp_path):
        out = tmp_path / "run"
        meta = await run_benchmark("cache", model, [_q("q1")], out, progress=lambda _m: None)

        assert meta.db == "isolated"
        assert json.loads((out / "run.json").read_text())["db"] == "isolated"
        assert load_run(out).meta.db == "isolated"
        note = meta.notes[0]
        assert "fresh temporary SQLite database" in note
        assert "per process" not in note and "in-process" not in note

    async def test_run_json_from_before_isolation_still_loads(self, default_db, model, tmp_path):
        out = tmp_path / "run"
        await run_benchmark("cache", model, [_q("q1")], out, progress=lambda _m: None)
        raw = json.loads((out / "run.json").read_text())
        del raw["db"]
        (out / "run.json").write_text(json.dumps(raw))
        assert load_run(out).meta.db is None


class TestCompareModelsIsolation:
    def test_each_models_run_gets_its_own_fresh_database(self, default_db, monkeypatch, tmp_path):
        paid = next(
            m for m in get_all_models()
            if m.provider == get_free_tier_model().provider and m.id != get_free_tier_model().id
        )
        for m in (get_free_tier_model(), paid):
            monkeypatch.setenv(provider_env_var(m), "fake-key-value-0123")
        monkeypatch.setattr("evals.benchmark.compare.make_judge_llm", lambda _model: FakeJudge())
        qfile = tmp_path / "questions.yaml"
        qfile.write_text(
            yaml.safe_dump(
                [
                    {"id": i, "ticker": "AAPL", "category": "filing_lookup", "question": f"Question {i}?"}
                    for i in ("q1", "q2")
                ]
            )
        )
        code = main(
            ["compare-models", "--graph", "cache", "--questions", str(qfile), "--out", str(tmp_path / "out"),
             "--models", paid.id, "--yes"]
        )

        assert code == 0
        # Reference run then the other model's run: both start empty. Without per-run
        # databases the second run would open with a hit.
        assert LOOKUPS == [("AAPL", False), ("AAPL", True), ("AAPL", False), ("AAPL", True)]
        for model_id in (get_free_tier_model().id, paid.id):
            assert load_run(tmp_path / "out" / "runs" / model_id).meta.db == "isolated"
        assert not default_db.exists()


class TestOtelTruncationWarning:
    TRUNCATION = "exceeds max length"

    def _truncation_records(self, caplog):
        return [r for r in caplog.records if r.name == OTEL_LOGGER and self.TRUNCATION in r.getMessage()]

    async def test_the_warning_is_real_without_the_cli(self, default_db, model, tmp_path, caplog):
        """Control: proves the CLI test below would catch the warning if it were not silenced."""
        with caplog.at_level(logging.WARNING):
            await run_benchmark("oversized", model, [_q("q1")], tmp_path / "run", progress=lambda _m: None)
        assert self._truncation_records(caplog)

    def test_the_cli_silences_it_during_a_run_and_restores_the_logger(
        self, default_db, monkeypatch, tmp_path, caplog
    ):
        monkeypatch.setenv(provider_env_var(get_free_tier_model()), "fake-key-value-0123")
        qfile = tmp_path / "questions.yaml"
        qfile.write_text(
            yaml.safe_dump([{"id": "q1", "ticker": "AAPL", "category": "filing_lookup", "question": "Question?"}])
        )
        before = logging.getLogger(OTEL_LOGGER).level
        with caplog.at_level(logging.WARNING):
            code = main(["run", "--graph", "oversized", "--questions", str(qfile), "--out", str(tmp_path / "run")])

        assert code == 0
        assert load_run(tmp_path / "run").questions["q1"].status == "ok"
        assert self._truncation_records(caplog) == []
        assert logging.getLogger(OTEL_LOGGER).level == before

    def test_the_cli_silences_it_during_compare_models_too(self, default_db, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv(provider_env_var(get_free_tier_model()), "fake-key-value-0123")
        monkeypatch.setattr("evals.benchmark.compare.make_judge_llm", lambda _model: FakeJudge())
        qfile = tmp_path / "questions.yaml"
        qfile.write_text(
            yaml.safe_dump([{"id": "q1", "ticker": "AAPL", "category": "filing_lookup", "question": "Question?"}])
        )
        with caplog.at_level(logging.WARNING):
            code = main(
                ["compare-models", "--graph", "oversized", "--questions", str(qfile),
                 "--out", str(tmp_path / "out"), "--yes"]
            )
        assert code == 0
        assert self._truncation_records(caplog) == []
