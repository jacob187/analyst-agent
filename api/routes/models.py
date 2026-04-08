"""Model registry endpoint — serves available LLM models to the frontend."""

import os

from fastapi import APIRouter

from agents.model_registry import get_all_models

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models():
    """Return all available LLM models with their metadata.

    No auth required — the model list is public configuration data
    used by the frontend to populate the model selector dropdown.
    """
    return {"models": [m.model_dump() for m in get_all_models()]}


@router.get("/env-keys")
async def env_keys():
    """Return which API keys are available from server environment variables.

    Returns booleans only — never exposes the actual key values.
    The frontend uses this to skip the settings prompt when keys are
    configured via .env for local development.
    """
    return {
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "sec_header": bool(os.getenv("SEC_HEADER")),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
    }
