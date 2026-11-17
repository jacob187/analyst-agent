"""Model registry endpoint — serves available LLM models to the frontend."""

import os

from fastapi import APIRouter, Depends

from agents.model_registry import get_all_models
from api.dependencies import (
    ANON_FREE_QUERIES,
    ApiKeys,
    free_trial_model_id,
    free_trial_open,
    get_api_keys,
    operator_may_lend,
)

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models():
    """Return all available LLM models with their metadata.

    No auth required — the model list is public configuration data
    used by the frontend to populate the model selector dropdown.
    """
    return {"models": [m.model_dump() for m in get_all_models()]}


@router.get("/env-keys")
async def env_keys(keys: ApiKeys = Depends(get_api_keys)):
    """Return which API keys are available to this caller from server env vars.

    Never the key values. Operator-key coverage is reported only to callers
    allowed to spend the operator's env keys (signed-in or self-host),
    mirroring the resolver's `_env_keys_allowed` gate: an anonymous caller sees
    nothing covered even when the operator has keys configured, so the UI never
    advertises a key the chat/filings path would then refuse. `is_operator_paid`
    reflects that gated resolution (no provider headers are sent to this
    endpoint, so an env key only resolves when the caller is allowed).

    Coverage is per model: `operator_models` lists the ids the operator's key
    pays for this caller (key resolved AND `operator_may_lend` — self-host
    lends every model). The per-provider booleans (`google`/`openai`/
    `anthropic`) mean "at least one of that provider's models is in
    `operator_models`", so a provider whose models are all non-lendable reads
    false. SEC_HEADER is a global server identity, reported to everyone.

    `free_trial_model_id` is reported to everyone, including anonymous
    callers, when the operator has configured a key for it and today's global
    trial cap isn't spent — it's the one deliberate exception to the gate
    above (see `api.dependencies.free_trial_key`).
    """
    trial_open = await free_trial_open()
    covered = [
        m for m in get_all_models()
        if keys.is_operator_paid(m.provider) and operator_may_lend(m)
    ]
    covered_providers = {m.provider for m in covered}
    return {
        "google": "google_genai" in covered_providers,
        "openai": "openai" in covered_providers,
        "anthropic": "anthropic" in covered_providers,
        "operator_models": [m.id for m in covered],
        "sec_header": bool(os.getenv("SEC_HEADER")),
        "tavily": bool(keys.tavily_api_key),
        "free_trial_model_id": free_trial_model_id() if trial_open else None,
        "free_trial_queries": ANON_FREE_QUERIES,
    }
