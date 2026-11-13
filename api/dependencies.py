"""Centralized API key resolution for all endpoints.

Keys are resolved with a consistent fallback chain:
  1. HTTP header (from browser localStorage via frontend)
  2. Environment variable / .env file (for local development)

This module provides FastAPI dependencies that any route can inject.
New endpoints should never read API keys directly — use these instead.

User identity gates persistence and operator-paid LLM access:
  - Anonymous (no user_id) → no DB writes and NO operator env keys. Anonymous
    visitors browse freely and may use LLM features only with their own (BYOK)
    keys sent per request.
  - With `CLERK_SECRET_KEY` configured the user id comes from the verified
    Clerk session token (REST: `X-Clerk-Session-Token`; WS:
    `clerk_session_token`). `X-User-Id` / `user_id` is optional and, if sent,
    must equal the token's subject. Verified users unlock DB persistence and
    may use the operator's env keys for `operator_lendable` models (bounded
    by the per-user daily budget); other models need their own key.

If Clerk is unconfigured (local dev / self-host) verification is skipped and
the single operator is trusted with env keys. `DISABLE_AUTH=true` does the same.

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
from typing import Any

from fastapi import Header, HTTPException

from agents.model_registry import ModelDef, get_free_tier_model
from api.db import get_llm_usage
from api.clerk_auth import is_auth_disabled, is_clerk_enabled, verify_clerk_token
from api.validators import USER_ID_RE

logger = logging.getLogger(__name__)


# Maps provider name → ApiKeys field name for key lookup.
_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "google_genai": "google_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}

# Maps provider name → the env var its operator key is read from.
PROVIDER_ENV_VARS: dict[str, str] = {
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# ── Anonymous free trial ─────────────────────────────────────────────────────
# A narrow, deliberate carve-out from `_env_keys_allowed`: anonymous visitors
# may spend the operator's key for ONE designated light model. The caller
# (api/routes/chat.py) caps it per IP per UTC day, plus a global daily cap
# across all anonymous traffic that bounds the operator's worst-case bill.
# Both counters live in `user_llm_usage` so they survive restarts/deploys.
# This exists so a demo visitor can chat without signing in or bringing a
# key; it does not change env-key resolution for any other model or route.
ANON_FREE_QUERIES = int(os.getenv("ANON_FREE_QUERIES", "3"))
ANON_FREE_DAILY_CAP = int(os.getenv("ANON_FREE_DAILY_CAP", "200"))
# Trial sessions also borrow the operator's Tavily key, metered per Tavily
# call (get_company_news's week fallback counts as a second). Basic searches
# are 1 credit, so 25 keeps a month inside Tavily's 1,000 free credits.
ANON_FREE_SEARCH_DAILY_CAP = int(os.getenv("ANON_FREE_SEARCH_DAILY_CAP", "25"))
# Pseudo user ids for `user_llm_usage` — can't collide with Clerk `user_...`
# ids or UUIDs.
FREE_TRIAL_GLOBAL_USAGE_KEY = "anon_trial:global"
FREE_TRIAL_SEARCH_USAGE_KEY = "anon_trial:search"


def free_trial_model_id() -> str | None:
    """The model anonymous visitors may use without a key, budget-capped.

    None when no model is flagged free_tier — the trial is disabled.
    """
    model = get_free_tier_model()
    return model.id if model else None


def free_trial_key() -> str | None:
    """The operator's env key for the free-trial model's provider, if any."""
    model = get_free_tier_model()
    if model is None:
        return None
    env_var = PROVIDER_ENV_VARS.get(model.provider)
    return os.getenv(env_var) if env_var else None


async def free_trial_open() -> bool:
    """The operator lends a trial key and today's global cap isn't spent."""
    return bool(free_trial_key()) and (
        await get_llm_usage(FREE_TRIAL_GLOBAL_USAGE_KEY) < ANON_FREE_DAILY_CAP
    )


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


def _is_clerk_user_id(user_id: str) -> bool:
    """True for Clerk account ids (`user_...`); False for anonymous UUIDs."""
    return user_id.startswith("user_")


def _trusted_user_id(user_id: str | None) -> str | None:
    """The identity trusted for persistence/scoping and operator resources.

    When Clerk is enabled, only Clerk-format ids are trustworthy; a missing id
    or a legacy UUID (which cannot be checked against Clerk) is anonymous —
    trusting an unverifiable id would let anyone claim any UUID's data. Clerk
    ids still require token verification by the caller (`_resolve_identity`)
    before they're acted on.
    """
    if is_auth_disabled() or not is_clerk_enabled():
        return user_id
    return user_id if (user_id and _is_clerk_user_id(user_id)) else None


def _env_keys_allowed(user_id: str | None) -> bool:
    """Whether the operator's env keys may resolve for this caller.

    Lent only to trusted callers: the local/self-host operator (auth disabled
    or Clerk unconfigured) or a verified signed-in user. Anonymous visitors
    resolve only their own BYOK keys — they never spend the operator's keys.
    """
    if is_auth_disabled() or not is_clerk_enabled():
        return True
    return _trusted_user_id(user_id) is not None


def operator_may_lend(model: ModelDef) -> bool:
    """Whether the operator's env key may pay for this model.

    With Clerk enabled (a hosted deploy) only models flagged
    `operator_lendable` in models.json run on the operator's key; any other
    model needs the caller's own key. Self-host / local dev (Clerk
    unconfigured or DISABLE_AUTH=true) is exempt: the env keys belong to the
    person running the server, so they may use any model with them.
    """
    if is_auth_disabled() or not is_clerk_enabled():
        return True
    return model.operator_lendable


def _resolve_identity(
    raw_user_id: str | None, token: str | None
) -> tuple[str | None, str | None]:
    """Resolve the caller's identity from the optional user id + Clerk token.

    Returns (user_id, error). `error` is a user-safe reason string when the
    credentials must be rejected (never contains token contents); otherwise
    None. Shared by the REST dependency and the WebSocket auth frame.

    With Clerk enabled the verified token is the single source of truth:
      - token present -> verified `sub` is the user id. A supplied user id is
        optional, but if it differs from `sub` the request is rejected.
      - invalid/expired token, no user id -> anonymous (tokens are short-lived;
        public endpoints must keep working). With a user id -> rejected.
      - no token, Clerk-format user id -> rejected (claim without proof).
      - no token, no id (or a legacy UUID, unverifiable against Clerk) ->
        anonymous (None); endpoint-level checks (require_user_id) then gate
        anything that needs a signed-in user.

    DISABLE_AUTH=true or Clerk unconfigured (local dev / self-host): the
    token is ignored and the validated user id is trusted as-is.
    """
    _maybe_warn_clerk_unconfigured()

    if is_auth_disabled() or not is_clerk_enabled():
        return _validate_user_id(raw_user_id), None

    if token:
        verified_sub = verify_clerk_token(token)
        if (
            verified_sub is None
            or not _is_clerk_user_id(verified_sub)
            or _validate_user_id(verified_sub) is None
        ):
            if raw_user_id:
                # An explicit identity claim that can't be verified.
                return None, "Invalid Clerk session token"
            # Clerk session tokens are short-lived: a visitor whose token just
            # expired must still reach public endpoints, as anonymous.
            logger.info("Clerk token failed verification; treating caller as anonymous")
            return None, None
        if raw_user_id and raw_user_id != verified_sub:
            logger.warning("Clerk token sub does not match supplied user_id")
            return None, "User ID does not match Clerk session"
        return verified_sub, None

    user_id = _validate_user_id(raw_user_id)
    if user_id and _is_clerk_user_id(user_id):
        return None, "Missing Clerk session token"
    return None, None


def _verify_user_identity(user_id: str | None, token: str | None) -> str | None:
    """REST wrapper over `_resolve_identity`: raises HTTPException(401) on rejection."""
    resolved, error = _resolve_identity(user_id, token)
    if error:
        raise HTTPException(status_code=401, detail=error)
    return resolved


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

    def key_for_model(self, model: ModelDef) -> str | None:
        """Return the key that may run `model`, or None.

        The caller's own (BYOK) key works for every model; the operator's
        env key only for models it may lend (`operator_may_lend`).
        """
        key = self.get_provider_key(model.provider)
        if key and self.is_operator_paid(model.provider) and not operator_may_lend(model):
            return None
        return key

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


def _resolve_key(
    user_supplied: str | None, env_var: str, allow_env: bool = True
) -> tuple[str | None, str | None]:
    """Resolve a key with provenance. Returns (key, source) where source is
    "user", "env", or None when no key is available. Env fallback is skipped
    when allow_env is False (anonymous callers never spend operator env keys)."""
    if user_supplied:
        return user_supplied, "user"
    if allow_env:
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
    user_id = _verify_user_identity(x_user_id, x_clerk_session_token)
    allow_env = _env_keys_allowed(user_id)
    google_key, google_source = _resolve_key(x_google_api_key, "GOOGLE_API_KEY", allow_env)
    openai_key, openai_source = _resolve_key(x_openai_api_key, "OPENAI_API_KEY", allow_env)
    anthropic_key, anthropic_source = _resolve_key(x_anthropic_api_key, "ANTHROPIC_API_KEY", allow_env)
    tavily_key, _ = _resolve_key(x_tavily_api_key, "TAVILY_API_KEY", allow_env)
    return ApiKeys(
        google_api_key=google_key,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        model_id=x_model_id or os.getenv("DEFAULT_MODEL_ID"),
        user_id=user_id,
        key_sources=_build_key_sources(google_source, openai_source, anthropic_source),
    )


_UNRESOLVED: Any = object()


def resolve_ws_keys(auth_message: dict, user_id: str | None = _UNRESOLVED) -> ApiKeys:
    """Resolve API keys from a WebSocket auth message -> env vars.

    WebSocket connections can't use HTTP headers, so the frontend sends
    keys in the first JSON message. Same fallback to env vars applies.

    Pass the id returned by `resolve_ws_identity` so the Clerk token is
    verified once; the route runs that first so it can send a structured
    error frame and close with policy-violation code. When omitted, identity
    is resolved here and a rejected identity degrades to anonymous.
    """
    if user_id is _UNRESOLVED:
        user_id, _error = resolve_ws_identity(auth_message)
    # The auth frame is untrusted JSON — a non-string model_id would blow up
    # registry lookups downstream.
    raw_model_id = auth_message.get("model_id")
    model_id = raw_model_id if isinstance(raw_model_id, str) else None
    allow_env = _env_keys_allowed(user_id)
    google_key, google_source = _resolve_key(auth_message.get("google_api_key"), "GOOGLE_API_KEY", allow_env)
    openai_key, openai_source = _resolve_key(auth_message.get("openai_api_key"), "OPENAI_API_KEY", allow_env)
    anthropic_key, anthropic_source = _resolve_key(auth_message.get("anthropic_api_key"), "ANTHROPIC_API_KEY", allow_env)
    tavily_key, _ = _resolve_key(auth_message.get("tavily_api_key"), "TAVILY_API_KEY", allow_env)
    return ApiKeys(
        google_api_key=google_key,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        model_id=model_id or os.getenv("DEFAULT_MODEL_ID"),
        user_id=user_id,
        key_sources=_build_key_sources(google_source, openai_source, anthropic_source),
    )


def resolve_ws_identity(auth_message: dict) -> tuple[str | None, str | None]:
    """Resolve the user id for a WebSocket auth frame.

    Returns (user_id, error_reason). With Clerk enabled the id comes from the
    verified `clerk_session_token`; the frame's optional `user_id` must match
    it. Errors are user-safe strings (no token contents).
    """
    token = auth_message.get("clerk_session_token")
    return _resolve_identity(
        auth_message.get("user_id"), token if isinstance(token, str) else None
    )
