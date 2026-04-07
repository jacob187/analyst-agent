"""Model registry endpoint — serves available LLM models to the frontend."""

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
