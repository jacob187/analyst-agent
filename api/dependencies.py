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
        llm = ChatGoogleGenerativeAI(google_api_key=keys.google_api_key, ...)
"""

import os
from dataclasses import dataclass

from fastapi import Header


@dataclass
class ApiKeys:
    """Resolved API keys available for the current request."""
    google_api_key: str | None
    sec_header: str | None
    tavily_api_key: str | None

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


async def get_api_keys(
    x_google_api_key: str | None = Header(None),
    x_sec_header: str | None = Header(None),
    x_tavily_api_key: str | None = Header(None),
) -> ApiKeys:
    """FastAPI dependency that resolves API keys from headers → env vars."""
    return ApiKeys(
        google_api_key=x_google_api_key or os.getenv("GOOGLE_API_KEY"),
        sec_header=x_sec_header or os.getenv("SEC_HEADER"),
        tavily_api_key=x_tavily_api_key or os.getenv("TAVILY_API_KEY"),
    )


def resolve_ws_keys(auth_message: dict) -> ApiKeys:
    """Resolve API keys from a WebSocket auth message → env vars.

    WebSocket connections can't use HTTP headers, so the frontend sends
    keys in the first JSON message. Same fallback to env vars applies.
    """
    return ApiKeys(
        google_api_key=auth_message.get("google_api_key") or os.getenv("GOOGLE_API_KEY"),
        sec_header=auth_message.get("sec_header") or os.getenv("SEC_HEADER"),
        tavily_api_key=auth_message.get("tavily_api_key") or os.getenv("TAVILY_API_KEY"),
    )
