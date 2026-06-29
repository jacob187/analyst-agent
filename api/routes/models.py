"""Model registry endpoint — serves available LLM models to the frontend."""

import os

from fastapi import APIRouter, Depends

from agents.model_registry import get_all_models
from api.dependencies import ApiKeys, get_api_keys

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

    Booleans only — never the key values. Provider keys are reported only to
    callers allowed to spend the operator's env keys (signed-in or self-host),
    mirroring the resolver's `_env_keys_allowed` gate: an anonymous caller sees
    false even when the operator has keys configured, so the UI never advertises
    a key the chat/filings path would then refuse. `is_operator_paid` reflects
    that gated resolution (no provider headers are sent to this endpoint, so an
    env key only resolves when the caller is allowed). SEC_HEADER is a global
    server identity, reported to everyone.
    """
    return {
        "google": keys.is_operator_paid("google_genai"),
        "openai": keys.is_operator_paid("openai"),
        "anthropic": keys.is_operator_paid("anthropic"),
        "sec_header": bool(os.getenv("SEC_HEADER")),
        "tavily": bool(keys.tavily_api_key),
    }
