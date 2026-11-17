"""Reading and writing run / judgement directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from evals.benchmark.common import BenchmarkError
from evals.benchmark.schemas import Judgement, QuestionResult, RunMeta

RUN_FILE = "run.json"
QUESTIONS_DIR = "questions"
JUDGEMENT_FILE = "judgement.json"


@dataclass
class LoadedRun:
    dir: Path
    meta: RunMeta
    questions: dict[str, QuestionResult]

    @property
    def label(self) -> str:
        return self.dir.name


def write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_model(path: Path, model_cls):
    try:
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BenchmarkError(f"Cannot read {path}: {e}") from e
    except ValidationError as e:
        raise BenchmarkError(f"{path} is not a valid {model_cls.__name__}: {e}") from e


def load_run(run_dir: Path | str) -> LoadedRun:
    run_dir = Path(run_dir)
    meta = _load_model(run_dir / RUN_FILE, RunMeta)
    questions: dict[str, QuestionResult] = {}
    for path in sorted((run_dir / QUESTIONS_DIR).glob("*.json")):
        result = _load_model(path, QuestionResult)
        questions[result.id] = result
    return LoadedRun(dir=run_dir, meta=meta, questions=questions)


def load_judgement(judgement_dir: Path | str) -> Judgement:
    return _load_model(Path(judgement_dir) / JUDGEMENT_FILE, Judgement)
