"""Question set: schema, loader and a stable content hash."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from api.validators import TICKER_RE
from evals.benchmark.common import BenchmarkError

DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"

Category = Literal["filing_lookup", "multi_step", "material_events", "technicals", "market_news"]
CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]

# Ids become file names (questions/<id>.json), so keep them path-safe (matched with
# fullmatch: ``$`` would also accept a trailing newline). They must also be unique
# ignoring case, since macOS and Windows file names are case-insensitive.
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class Question(BaseModel):
    id: str
    ticker: str
    category: Category
    question: str

    @field_validator("id")
    @classmethod
    def _id_is_path_safe(cls, v: str) -> str:
        if not _ID_RE.fullmatch(v):
            raise ValueError(f"id {v!r} must match {_ID_RE.pattern}")
        return v

    @field_validator("ticker")
    @classmethod
    def _ticker_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.fullmatch(v):
            raise ValueError(f"invalid ticker {v!r}")
        return v

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question text is empty")
        return v


def load_questions(path: Path | str = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    """Load and validate a question file (a YAML list, or ``{questions: [...]}``)."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BenchmarkError(f"Cannot read question file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise BenchmarkError(f"Question file {path} is not valid YAML: {e}") from e

    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list) or not data:
        raise BenchmarkError(f"Question file {path} must contain a non-empty list of questions.")

    try:
        questions = [Question.model_validate(item) for item in data]
    except ValidationError as e:
        raise BenchmarkError(f"Invalid question in {path}: {e}") from e

    seen: dict[str, str] = {}
    for q in questions:
        key = q.id.casefold()
        if key in seen:
            same = "" if seen[key] == q.id else f" (same file name as {seen[key]!r}: ids must differ in more than case)"
            raise BenchmarkError(f"Duplicate question id {q.id!r} in {path}{same}.")
        seen[key] = q.id
    return questions


def question_set_hash(questions: list[Question]) -> str:
    """SHA-256 over the canonical JSON of the questions (insensitive to YAML formatting)."""
    canonical = json.dumps(
        [q.model_dump() for q in questions], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
