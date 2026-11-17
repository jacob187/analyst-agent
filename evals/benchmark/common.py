"""Shared plumbing: error type, model resolution, provider keys, environment setup.

No model id is hardcoded anywhere in this package. A missing ``--model`` means
the registry's free-tier model.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from agents.model_registry import ModelDef, get_all_models, get_free_tier_model, get_model

DEFAULT_TIMEOUT_S = 600.0
# Longest error text written into a run or judgement file.
ERROR_TEXT_LIMIT = 500

# Env vars whose values must never appear in written output.
_SECRET_ENV_VARS = ("TAVILY_API_KEY",)
_MIN_SECRET_LEN = 8


class BenchmarkError(Exception):
    """A user-facing benchmark failure (bad flag, missing key, bad input)."""


def utc_now() -> str:
    """The timestamp written into run, judgement and summary files."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rate(numerator: float, n: int) -> float | None:
    """``numerator / n`` rounded to 4 places; None for an empty denominator."""
    return None if n == 0 else round(numerator / n, 4)


def describe_error(exc: BaseException) -> str:
    """``Type: message`` for a run or judgement file: secrets scrubbed, length capped."""
    text = str(exc) or "(no message)"
    return scrub_secrets(f"{type(exc).__name__}: {text}")[:ERROR_TEXT_LIMIT]


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


def has_provider_key(model: ModelDef) -> bool:
    """Whether the environment holds ``model``'s provider key (the value is not exposed)."""
    try:
        provider_api_key(model)
    except BenchmarkError:
        return False
    return True


def prepare_environment() -> None:
    """What ``api/main.py`` does at startup that a bare CLI process lacks.

    Loads the project ``.env`` into the process environment (as the app does) and
    sets the SEC EDGAR identity edgartools requires through the app's own
    ``set_sec_identity`` (same ``SEC_HEADER`` fallback). Importing ``api.main`` also
    applies the app's logging setup (``LOG_LEVEL``, default INFO) to the CLI.
    """
    from dotenv import load_dotenv

    load_dotenv()
    from api.main import set_sec_identity  # after .env is loaded: the import reads LOG_LEVEL

    set_sec_identity()


def scrub_secrets(text: str) -> str:
    """Replace any configured API key found in ``text`` with ``[redacted]``."""
    from api.dependencies import PROVIDER_ENV_VARS

    for name in (*PROVIDER_ENV_VARS.values(), *_SECRET_ENV_VARS):
        value = (os.getenv(name) or "").strip()
        if len(value) >= _MIN_SECRET_LEN:
            text = text.replace(value, "[redacted]")
    return text
