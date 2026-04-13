"""Centralized API key resolution for all endpoints.

Keys are resolved with a consistent fallback chain:
  1. HTTP header (from browser localStorage via frontend)
  2. Environment variable / .env file (for local development)

This module provides FastAPI dependencies that any route can inject.
New endpoints should never read API keys directly — use these instead.

Usage in a route:
    from api.dependencies import ApiKeys, get_api_keys

    @router.get("/example")
    async def example(keys: ApiKeys = Depends(get_api_keys)):
        api_key = keys.require_provider_key(model.provider)
        llm = create_llm(model_id, api_key)
"""

import os
from dataclasses import dataclass

from fastapi import Header

from api.validators import USER_ID_RE


# Maps provider name → ApiKeys field name for key lookup.
_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "google_genai": "google_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


def _validate_user_id(raw: str | None) -> str | None:
    """Return the user ID if it's a valid UUID, otherwise None."""
    if raw and isinstance(raw, str) and USER_ID_RE.match(raw):
        return raw
    return None


@dataclass
class ApiKeys:
    """Resolved API keys available for the current request."""
    google_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    sec_header: str | None
    tavily_api_key: str | None
    model_id: str | None
    user_id: str | None = None

    def require_user_id(self) -> str:
        """Return user_id or raise ValueError if missing/invalid."""
        if not self.user_id:
            raise ValueError("User ID required (X-User-Id header)")
        return self.user_id

    def require_google(self) -> str:
        """Return google_api_key or raise ValueError if missing."""
        if not self.google_api_key:
            raise ValueError("Google API key required")
        return self.google_api_key

    def require_sec(self) -> str:
        """Return sec_header or raise ValueError if missing."""
        if not self.sec_header:
            raise ValueError("SEC header required")
        return self.sec_header

    def get_provider_key(self, provider: str) -> str | None:
        """Return the API key for a provider, or None if not set."""
        field = _PROVIDER_KEY_FIELDS.get(provider)
        if field is None:
            return None
        return getattr(self, field, None)

    def require_provider_key(self, provider: str) -> str:
        """Return the API key for a provider, or raise ValueError."""
        key = self.get_provider_key(provider)
        if not key:
            provider_display = provider.replace("_", " ").title()
            raise ValueError(f"{provider_display} API key required")
        return key


async def get_api_keys(
    x_google_api_key: str | None = Header(None),
    x_openai_api_key: str | None = Header(None),
    x_anthropic_api_key: str | None = Header(None),
    x_sec_header: str | None = Header(None),
    x_tavily_api_key: str | None = Header(None),
    x_model_id: str | None = Header(None),
    x_user_id: str | None = Header(None),
) -> ApiKeys:
    """FastAPI dependency that resolves API keys from headers → env vars."""
    return ApiKeys(
        google_api_key=x_google_api_key or os.getenv("GOOGLE_API_KEY"),
        openai_api_key=x_openai_api_key or os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=x_anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        sec_header=x_sec_header or os.getenv("SEC_HEADER"),
        tavily_api_key=x_tavily_api_key or os.getenv("TAVILY_API_KEY"),
        model_id=x_model_id,
        user_id=_validate_user_id(x_user_id),
    )


def resolve_ws_keys(auth_message: dict) -> ApiKeys:
    """Resolve API keys from a WebSocket auth message → env vars.

    WebSocket connections can't use HTTP headers, so the frontend sends
    keys in the first JSON message. Same fallback to env vars applies.
    """
    return ApiKeys(
        google_api_key=auth_message.get("google_api_key") or os.getenv("GOOGLE_API_KEY"),
        openai_api_key=auth_message.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=auth_message.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"),
        sec_header=auth_message.get("sec_header") or os.getenv("SEC_HEADER"),
        tavily_api_key=auth_message.get("tavily_api_key") or os.getenv("TAVILY_API_KEY"),
        model_id=auth_message.get("model_id"),
        user_id=_validate_user_id(auth_message.get("user_id")),
    )
