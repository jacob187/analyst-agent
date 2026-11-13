"""Shared plumbing: error type, model resolution, provider keys, environment setup.

No model id is hardcoded anywhere in this package. A missing ``--model`` means
the registry's free-tier model.
"""

from __future__ import annotations

import os

from agents.model_registry import ModelDef, get_all_models, get_free_tier_model, get_model

DEFAULT_TIMEOUT_S = 600.0

# Env vars whose values must never appear in written output.
_SECRET_ENV_VARS = ("TAVILY_API_KEY",)
_MIN_SECRET_LEN = 8


class BenchmarkError(Exception):
    """A user-facing benchmark failure (bad flag, missing key, bad input)."""


def resolve_model(model_id: str | None) -> ModelDef:
    """Registry model for ``--model``; the free-tier model when omitted.

    Raises BenchmarkError for an unknown id or when no free-tier model exists.
    """
    if model_id is None:
        model = get_free_tier_model()
        if model is None:
            raise BenchmarkError(
                "No --model given and the registry has no free-tier model; pass --model explicitly."
            )
        return model
    model = get_model(model_id)
    if model is None:
        known = ", ".join(sorted(m.id for m in get_all_models()))
        raise BenchmarkError(f"Unknown model {model_id!r}. Registry models: {known}")
    return model


def provider_env_var(model: ModelDef) -> str:
    from api.dependencies import PROVIDER_ENV_VARS

    env_var = PROVIDER_ENV_VARS.get(model.provider)
    if not env_var:
        raise BenchmarkError(f"No API key env var is known for provider {model.provider!r}.")
    return env_var


def provider_api_key(model: ModelDef) -> str:
    """The provider key for ``model`` from the environment (never printed)."""
    env_var = provider_env_var(model)
    key = (os.getenv(env_var) or "").strip()
    if not key:
        raise BenchmarkError(
            f"{env_var} is not set; it is required to run model {model.id!r}."
        )
    return key


def prepare_environment() -> None:
    """What ``api/main.py`` does at startup that a bare CLI process lacks.

    Loads the project ``.env`` into the process environment (as the app does)
    and sets the SEC EDGAR identity edgartools requires.
    """
    from dotenv import load_dotenv
    from edgar import set_identity

    load_dotenv()
    set_identity((os.getenv("SEC_HEADER") or "").strip() or "Analyst Agent dev@localhost")


def scrub_secrets(text: str) -> str:
    """Replace any configured API key found in ``text`` with ``[redacted]``."""
    from api.dependencies import PROVIDER_ENV_VARS

    for name in (*PROVIDER_ENV_VARS.values(), *_SECRET_ENV_VARS):
        value = (os.getenv(name) or "").strip()
        if len(value) >= _MIN_SECRET_LEN:
            text = text.replace(value, "[redacted]")
    return text
