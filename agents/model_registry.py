"""Model registry — single source of truth for available LLM models.

Loads model definitions from models.json at import time and exposes
lookup functions used by the LLM factory, memory compression, and
the GET /models API endpoint.

Adding a new model requires only editing models.json — no code changes.
"""

import json
from pathlib import Path

from pydantic import BaseModel


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


# ---------------------------------------------------------------------------
# Module-level loading — fails fast if models.json is missing or malformed.
# ---------------------------------------------------------------------------

_MODELS_PATH = Path(__file__).parent / "models.json"
_raw = json.loads(_MODELS_PATH.read_text())
_MODELS: dict[str, ModelDef] = {
    entry["id"]: ModelDef(**entry) for entry in _raw
}

# Validate exactly one default model exists.
_defaults = [m for m in _MODELS.values() if m.default]
if len(_defaults) != 1:
    raise ValueError(
        f"models.json must have exactly one default model, found {len(_defaults)}"
    )

_DEFAULT_MODEL: ModelDef = _defaults[0]

# Fallback threshold when a model isn't in the registry (e.g. custom model ID).
_FALLBACK_THRESHOLD = 30_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model(model_id: str) -> ModelDef | None:
    """Return the model definition for an ID, or None if not found."""
    return _MODELS.get(model_id)


def get_default_model() -> ModelDef:
    """Return the single default model."""
    return _DEFAULT_MODEL


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
