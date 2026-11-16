"""AC1 (question set schema) and AC6 (no hardcoded model ids, default resolution)."""

from pathlib import Path

import pytest

import agents.model_registry as registry
import evals.benchmark as benchmark_pkg
from evals.benchmark.common import BenchmarkError, resolve_model
from evals.benchmark.questions import (
    CATEGORIES,
    DEFAULT_QUESTIONS_PATH,
    Question,
    load_questions,
    question_set_hash,
)

pytestmark = pytest.mark.eval_unit


class TestShippedQuestionSet:
    def test_at_least_20_questions_across_3_tickers(self):
        questions = load_questions()
        assert len(questions) >= 20
        assert len({q.ticker for q in questions}) >= 3

    def test_every_category_has_at_least_3_questions(self):
        questions = load_questions()
        assert set(CATEGORIES) == {
            "filing_lookup", "multi_step", "material_events", "technicals", "market_news",
        }
        for category in CATEGORIES:
            assert sum(q.category == category for q in questions) >= 3, category

    def test_ids_unique_and_fields_present(self):
        questions = load_questions()
        assert len({q.id for q in questions}) == len(questions)
        for q in questions:
            assert q.ticker and q.question and q.id and q.category

    def test_hash_is_stable_and_content_sensitive(self):
        questions = load_questions()
        assert question_set_hash(questions) == question_set_hash(load_questions())
        edited = [*questions[:-1], questions[-1].model_copy(update={"question": "Something else?"})]
        assert question_set_hash(edited) != question_set_hash(questions)


class TestLoaderRejectsBadFiles:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "q.yaml"
        path.write_text(text)
        return path

    def test_unknown_category(self, tmp_path):
        path = self._write(tmp_path, "- {id: a, ticker: AAPL, category: gossip, question: Why}\n")
        with pytest.raises(BenchmarkError, match="Invalid question"):
            load_questions(path)

    def test_duplicate_ids(self, tmp_path):
        row = "- {id: a, ticker: AAPL, category: technicals, question: Why}\n"
        with pytest.raises(BenchmarkError, match="Duplicate question id"):
            load_questions(self._write(tmp_path, row * 2))

    def test_path_unsafe_id(self, tmp_path):
        path = self._write(tmp_path, "- {id: ../evil, ticker: AAPL, category: technicals, question: Why}\n")
        with pytest.raises(BenchmarkError, match="Invalid question"):
            load_questions(path)

    def test_missing_field(self, tmp_path):
        with pytest.raises(BenchmarkError, match="Invalid question"):
            load_questions(self._write(tmp_path, "- {id: a, ticker: AAPL, category: technicals}\n"))

    def test_empty_file_and_missing_file(self, tmp_path):
        with pytest.raises(BenchmarkError, match="non-empty list"):
            load_questions(self._write(tmp_path, "[]\n"))
        with pytest.raises(BenchmarkError, match="Cannot read"):
            load_questions(tmp_path / "absent.yaml")

    @pytest.mark.parametrize("id_yaml", ['"nvda-risk\\n"', "|\n    nvda-risk"])
    def test_an_id_with_a_trailing_newline_is_rejected(self, tmp_path, id_yaml):
        path = self._write(tmp_path, f"- id: {id_yaml}\n  ticker: NVDA\n  category: technicals\n  question: Why\n")
        with pytest.raises(BenchmarkError, match="Invalid question"):
            load_questions(path)

    def test_ids_that_differ_only_in_case_are_duplicates(self, tmp_path):
        # Ids become file names, and macOS / Windows file names are case-insensitive.
        rows = (
            "- {id: Q1, ticker: AAPL, category: technicals, question: Why}\n"
            "- {id: q1, ticker: AAPL, category: technicals, question: How}\n"
        )
        with pytest.raises(BenchmarkError, match="Duplicate question id 'q1'.*'Q1'"):
            load_questions(self._write(tmp_path, rows))

    def test_tickers_follow_the_apps_ticker_rule(self):
        assert Question(id="a", ticker="1abc", category="technicals", question="Why?").ticker == "1ABC"
        with pytest.raises(ValueError, match="invalid ticker"):
            Question(id="a", ticker="AAPL/X", category="technicals", question="Why?")

    def test_ticker_is_normalised(self):
        q = Question(id="a", ticker=" aapl ", category="technicals", question="Why?")
        assert q.ticker == "AAPL"


class TestNoHardcodedModelIds:
    def test_no_registry_model_id_appears_in_the_package(self):
        package_dir = Path(benchmark_pkg.__file__).parent
        files = [
            p for p in package_dir.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".yml", ".md", ".json"}
        ]
        assert DEFAULT_QUESTIONS_PATH in files
        assert any(p.name == "judge.yaml" for p in files)
        model_ids = [m.id for m in registry.get_all_models()]
        assert model_ids
        offenders = [
            (p.name, model_id)
            for p in files
            for model_id in model_ids
            if model_id in p.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestModelResolution:
    def test_default_is_the_free_tier_model(self, monkeypatch):
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        free = registry.get_free_tier_model()
        assert free is not None
        assert resolve_model(None).id == free.id

    def test_explicit_registry_model_is_returned(self):
        model = registry.get_all_models()[0]
        assert resolve_model(model.id) == model

    def test_unknown_model_is_rejected_with_the_known_ids(self):
        with pytest.raises(BenchmarkError, match="Unknown model 'no-such-model'") as exc:
            resolve_model("no-such-model")
        assert registry.get_all_models()[0].id in str(exc.value)

    def test_no_free_tier_model_and_no_flag_is_an_error(self, monkeypatch):
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.setattr(registry, "_FREE_TIER_MODEL", None)
        with pytest.raises(BenchmarkError, match="no free-tier model"):
            resolve_model(None)
