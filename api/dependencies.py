"""Centralized API key resolution for all endpoints.

Keys are resolved with a consistent fallback chain:
  1. HTTP header (from browser localStorage via frontend)
  2. Environment variable / .env file (for local development)

This module provides FastAPI dependencies that any route can inject.
New endpoints should never read API keys directly — use these instead.

User identity is verified via Clerk when `CLERK_SECRET_KEY` is configured:
  - REST: clients send `X-Clerk-Session-Token` alongside `X-User-Id`.
  - WS:   clients send `clerk_session_token` alongside `user_id`.

If Clerk is unconfigured (local dev / self-host) verification is skipped
entirely. `DISABLE_AUTH=true` always skips verification.

Usage in a route:
    from api.dependencies import ApiKeys, get_api_keys

    @router.get("/example")
    async def example(keys: ApiKeys = Depends(get_api_keys)):
        api_key = keys.require_provider_key(model.provider)
        llm = create_llm(model_id, api_key)
"""

import logging
import os
from dataclasses import dataclass, field

from fastapi import Header, HTTPException

from api.clerk_auth import is_auth_disabled, is_clerk_enabled, verify_clerk_token
from api.validators import USER_ID_RE

logger = logging.getLogger(__name__)


# Maps provider name → ApiKeys field name for key lookup.
_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "google_genai": "google_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


_clerk_unconfigured_warned = False


def _maybe_warn_clerk_unconfigured() -> None:
    global _clerk_unconfigured_warned
    if _clerk_unconfigured_warned:
        return
    if is_auth_disabled():
        logger.warning("DISABLE_AUTH=true — Clerk verification is bypassed for all requests")
    elif not is_clerk_enabled():
        logger.warning(
            "CLERK_SECRET_KEY not set — running without Clerk verification "
            "(acceptable for local dev / self-host; do NOT deploy to production)"
        )
    _clerk_unconfigured_warned = True


def _validate_user_id(raw: str | None) -> str | None:
    """Return the user ID if it matches an accepted format, otherwise None."""
    if raw and isinstance(raw, str) and USER_ID_RE.match(raw):
        return raw
    return None


def _verify_user_identity(user_id: str | None, token: str | None) -> str | None:
    """Cross-check user_id against a Clerk session token.

    Returns the verified user_id, or raises HTTPException(401) on mismatch
    when Clerk is enabled.

    Behavior:
      - DISABLE_AUTH=true → skip; trust the validated user_id as-is.
      - Clerk unconfigured → skip; trust the validated user_id as-is.
      - Clerk configured + no token → reject (401) when user_id is present.
      - Clerk configured + token → verify; sub must equal user_id.
    """
    _maybe_warn_clerk_unconfigured()

    if is_auth_disabled() or not is_clerk_enabled():
        return user_id

    if not user_id:
        # No user_id supplied; nothing to verify. Endpoint-level checks
        # (require_user_id) decide whether that's acceptable.
        return None

    if not token:
        raise HTTPException(status_code=401, detail="Missing Clerk session token")

    verified_sub = verify_clerk_token(token)
    if verified_sub is None:
        raise HTTPException(status_code=401, detail="Invalid Clerk session token")
    if verified_sub != user_id:
        logger.warning("Clerk token sub does not match supplied user_id")
        raise HTTPException(status_code=401, detail="User ID does not match Clerk session")

    return user_id


@dataclass
class ApiKeys:
    """Resolved API keys available for the current request."""
    google_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    tavily_api_key: str | None
    model_id: str | None
    user_id: str | None = None
    # Per-provider key provenance — "user" if the resolved key came from a
    # request header / WS auth message, "env" if it came from server env.
    # Absent if the provider's key is None. Used by the LLM budget check
    # to skip BYOK requests (operator only pays for env-keyed traffic).
    key_sources: dict[str, str] = field(default_factory=dict)

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

    def get_provider_key(self, provider: str) -> str | None:
        """Return the API key for a provider, or None if not set."""
        field_name = _PROVIDER_KEY_FIELDS.get(provider)
        if field_name is None:
            return None
        return getattr(self, field_name, None)

    def require_provider_key(self, provider: str) -> str:
        """Return the API key for a provider, or raise ValueError."""
        key = self.get_provider_key(provider)
        if not key:
            provider_display = provider.replace("_", " ").title()
            raise ValueError(f"{provider_display} API key required")
        return key

    def is_operator_paid(self, provider: str) -> bool:
        """True iff the resolved provider key came from server env.

        Used by the per-user LLM budget check to skip BYOK requests —
        when a user supplies their own key in headers / WS auth, the
        operator isn't paying so the budget doesn't apply.
        """
        return self.key_sources.get(provider) == "env"


def _resolve_key(user_supplied: str | None, env_var: str) -> tuple[str | None, str | None]:
    """Resolve a key with provenance. Returns (key, source) where source is
    "user", "env", or None when no key is available."""
    if user_supplied:
        return user_supplied, "user"
    env_value = os.getenv(env_var)
    if env_value:
        return env_value, "env"
    return None, None


def _build_key_sources(
    google_source: str | None,
    openai_source: str | None,
    anthropic_source: str | None,
) -> dict[str, str]:
    sources: dict[str, str] = {}
    if google_source:
        sources["google_genai"] = google_source
    if openai_source:
        sources["openai"] = openai_source
    if anthropic_source:
        sources["anthropic"] = anthropic_source
    return sources


async def get_api_keys(
    x_google_api_key: str | None = Header(None),
    x_openai_api_key: str | None = Header(None),
    x_anthropic_api_key: str | None = Header(None),
    x_tavily_api_key: str | None = Header(None),
    x_model_id: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_clerk_session_token: str | None = Header(None),
) -> ApiKeys:
    """FastAPI dependency that resolves API keys from headers → env vars."""
    user_id = _validate_user_id(x_user_id)
    user_id = _verify_user_identity(user_id, x_clerk_session_token)
    google_key, google_source = _resolve_key(x_google_api_key, "GOOGLE_API_KEY")
    openai_key, openai_source = _resolve_key(x_openai_api_key, "OPENAI_API_KEY")
    anthropic_key, anthropic_source = _resolve_key(x_anthropic_api_key, "ANTHROPIC_API_KEY")
    tavily_key, _ = _resolve_key(x_tavily_api_key, "TAVILY_API_KEY")
    return ApiKeys(
        google_api_key=google_key,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        model_id=x_model_id or os.getenv("DEFAULT_MODEL_ID"),
        user_id=user_id,
        key_sources=_build_key_sources(google_source, openai_source, anthropic_source),
    )


def resolve_ws_keys(auth_message: dict) -> ApiKeys:
    """Resolve API keys from a WebSocket auth message → env vars.

    WebSocket connections can't use HTTP headers, so the frontend sends
    keys in the first JSON message. Same fallback to env vars applies.

    Clerk verification is NOT performed here — the WebSocket route runs
    `verify_ws_identity` separately so it can send a structured error
    frame and close with policy-violation code.
    """
    google_key, google_source = _resolve_key(auth_message.get("google_api_key"), "GOOGLE_API_KEY")
    openai_key, openai_source = _resolve_key(auth_message.get("openai_api_key"), "OPENAI_API_KEY")
    anthropic_key, anthropic_source = _resolve_key(auth_message.get("anthropic_api_key"), "ANTHROPIC_API_KEY")
    tavily_key, _ = _resolve_key(auth_message.get("tavily_api_key"), "TAVILY_API_KEY")
    return ApiKeys(
        google_api_key=google_key,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        model_id=auth_message.get("model_id") or os.getenv("DEFAULT_MODEL_ID"),
        user_id=_validate_user_id(auth_message.get("user_id")),
        key_sources=_build_key_sources(google_source, openai_source, anthropic_source),
    )


def verify_ws_identity(user_id: str | None, auth_message: dict) -> tuple[bool, str | None]:
    """Verify a WebSocket auth message's user_id against its Clerk token.

    Returns (ok, error_reason). When Clerk is disabled, always returns
    (True, None). Errors are user-safe strings (no token contents).
    """
    _maybe_warn_clerk_unconfigured()

    if is_auth_disabled() or not is_clerk_enabled():
        return True, None
    if not user_id:
        return True, None  # downstream code decides if user_id is required

    token = auth_message.get("clerk_session_token")
    if not token:
        return False, "Missing Clerk session token"

    verified_sub = verify_clerk_token(token)
    if verified_sub is None:
        return False, "Invalid Clerk session token"
    if verified_sub != user_id:
        logger.warning("WS Clerk token sub does not match supplied user_id")
        return False, "User ID does not match Clerk session"
    return True, None
