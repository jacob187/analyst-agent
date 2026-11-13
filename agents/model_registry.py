"""Model registry — single source of truth for available LLM models.

Loads model definitions from models.json at import time and exposes
lookup functions used by the LLM factory, memory compression, and
the GET /models API endpoint.

Adding a new model requires only editing models.json — no code changes.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, StrictBool, StrictFloat, field_validator

# Non-negative USD per million tokens. Strict: a quoted "3" or a bool is a
# models.json typo, not a price.
_Price = Annotated[StrictFloat, Field(ge=0)]


class ModelDef(BaseModel):
    """A supported LLM model definition.

    Fields map directly to models.json entries. The model is frozen
    so instances can be safely shared across threads/requests.
    """

    model_config = {"frozen": True}

    id: str
    provider: str
    display_name: str
    max_context: int
    thinking_capable: bool
    default: bool
    free_tier: bool = False
    # Whether a signed-in user without their own key may run this model on
    # the operator's env key (see api.dependencies.operator_may_lend).
    operator_lendable: StrictBool = False
    input_price_per_mtok: _Price | None = None
    output_price_per_mtok: _Price | None = None
    # ISO date ("YYYY-MM-DD") the prices were last verified.
    prices_as_of: str | None = None

    @field_validator("prices_as_of")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("prices_as_of must be YYYY-MM-DD")
            date.fromisoformat(value)  # rejects impossible dates (2026-13-01)
        return value


# ---------------------------------------------------------------------------
# Module-level loading — fails fast if models.json is missing or malformed.
# ---------------------------------------------------------------------------


def _build_registry(
    raw: list[dict[str, Any]],
) -> tuple[dict[str, ModelDef], ModelDef, ModelDef | None]:
    """Parse and validate models.json entries.

    Returns (models by id, the default model, the free-tier model or None).
    Raises ValueError (pydantic's ValidationError included) on bad data.
    """
    models = {entry["id"]: ModelDef(**entry) for entry in raw}

    # Validate exactly one default model exists.
    defaults = [m for m in models.values() if m.default]
    if len(defaults) != 1:
        raise ValueError(
            f"models.json must have exactly one default model, found {len(defaults)}"
        )

    # At most one model may be flagged as the free-tier model — anonymous
    # visitors are only ever lent a key for one specific model.
    free_tier_models = [m for m in models.values() if m.free_tier]
    if len(free_tier_models) > 1:
        raise ValueError(
            f"models.json must have at most one free_tier model, found {len(free_tier_models)}"
        )
    # Anonymous visitors get the free-tier model on the operator's key, so
    # signed-in users must be able to as well.
    for m in free_tier_models:
        if not m.operator_lendable:
            raise ValueError(f"free_tier model {m.id!r} must be operator_lendable")

    return models, defaults[0], (free_tier_models[0] if free_tier_models else None)


_MODELS_PATH = Path(__file__).parent / "models.json"
_MODELS, _DEFAULT_MODEL, _FREE_TIER_MODEL = _build_registry(
    json.loads(_MODELS_PATH.read_text())
)

# Fallback threshold when a model isn't in the registry (e.g. custom model ID).
_FALLBACK_THRESHOLD = 30_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model(model_id: str) -> ModelDef | None:
    """Return the model definition for an ID, or None if not found."""
    return _MODELS.get(model_id)


def get_default_model() -> ModelDef:
    """Return the default model — env var DEFAULT_MODEL_ID overrides models.json."""
    import os
    env_id = os.getenv("DEFAULT_MODEL_ID")
    if env_id:
        model = _MODELS.get(env_id)
        if model:
            return model
    return _DEFAULT_MODEL


def get_free_tier_model() -> ModelDef | None:
    """Return the model anonymous free-trial callers may use, or None.

    env var ANON_FREE_MODEL_ID overrides models.json. Returns None when no
    model is flagged free_tier — the trial is off. Falling back to the default
    model instead would silently lend the operator's key for whatever model
    the default happens to be.
    """
    import os
    env_id = os.getenv("ANON_FREE_MODEL_ID")
    if env_id:
        model = _MODELS.get(env_id)
        if model:
            return model
    return _FREE_TIER_MODEL


def get_all_models() -> list[ModelDef]:
    """Return all registered models."""
    return list(_MODELS.values())


def get_models_by_provider(provider: str) -> list[ModelDef]:
    """Return models filtered by provider (e.g. 'google_genai', 'openai')."""
    return [m for m in _MODELS.values() if m.provider == provider]


def get_token_threshold(model_id: str) -> int:
    """Return the token compression threshold for a model.

    Uses ~25% of the model's context window — leaves room for system prompt,
    tools, and the model's response. Falls back to 30k for unknown models.
    """
    model = _MODELS.get(model_id)
    if model is None:
        return _FALLBACK_THRESHOLD
    return model.max_context // 4
