"""Clerk JWT verification using Clerk's published JWKS.

Verifies session tokens issued by Clerk by:
  1. Fetching Clerk's JWKS (cached in-memory, refreshed every ~10 minutes).
  2. Locating the signing key by `kid`.
  3. Verifying signature (RS256), `exp`, and `iss` claims.

We use raw JWT verification rather than `clerk-backend-api.authenticate_request`
because that helper requires an `httpx.Request` object, which is awkward for
WebSocket auth (we only have a token string). Direct verification works
uniformly across REST and WebSocket contexts.

Configure via env:
  CLERK_SECRET_KEY  — presence of this enables verification (any non-empty value)
  CLERK_ISSUER_URL  — Clerk frontend API origin, e.g. https://your-app.clerk.accounts.dev
  CLERK_JWKS_URL    — optional override; defaults to {issuer}/.well-known/jwks.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError

logger = logging.getLogger(__name__)

_JWKS_TTL_SECONDS = 10 * 60  # refresh every 10 minutes

# Module-level cache: (fetched_at_epoch, PyJWKClient instance) keyed by JWKS URL.
_jwks_cache: dict[str, tuple[float, PyJWKClient]] = {}


def is_clerk_enabled() -> bool:
    return bool(os.getenv("CLERK_SECRET_KEY"))


def is_auth_disabled() -> bool:
    return (os.getenv("DISABLE_AUTH") or "").lower() in ("1", "true", "yes")


def _issuer() -> str | None:
    issuer = os.getenv("CLERK_ISSUER_URL")
    if not issuer:
        return None
    return issuer.rstrip("/")


def _jwks_url() -> str | None:
    override = os.getenv("CLERK_JWKS_URL")
    if override:
        return override
    issuer = _issuer()
    if not issuer:
        return None
    return f"{issuer}/.well-known/jwks.json"


def _get_jwk_client(jwks_url: str) -> PyJWKClient:
    """Return a cached PyJWKClient for the URL, refreshing after the TTL."""
    cached = _jwks_cache.get(jwks_url)
    now = time.time()
    if cached and (now - cached[0]) < _JWKS_TTL_SECONDS:
        return cached[1]
    client = PyJWKClient(jwks_url, cache_keys=True, lifespan=_JWKS_TTL_SECONDS)
    _jwks_cache[jwks_url] = (now, client)
    return client


def _reset_cache_for_tests() -> None:
    _jwks_cache.clear()


class ClerkVerifierUnavailable(RuntimeError):
    """The token can't be checked at all: verification is misconfigured or
    Clerk's JWKS is unreachable. Distinct from an invalid/expired token."""


def verify_clerk_token(token: str) -> str | None:
    """Verify a Clerk session JWT and return the `sub` claim, or None on failure.

    Returns None for a token that fails verification (expired, bad signature,
    wrong issuer, unknown key). Raises `ClerkVerifierUnavailable` when the
    verifier itself can't run (CLERK_ISSUER_URL unset, JWKS fetch failed), so
    callers don't mistake an outage for a signed-out user.

    Never logs the token itself. Logs token failures at WARN level and
    verifier faults at ERROR.
    """
    if not token or not isinstance(token, str):
        return None

    issuer = _issuer()
    jwks_url = _jwks_url()
    if not issuer or not jwks_url:
        logger.error("Clerk verification requested but CLERK_ISSUER_URL is not set")
        raise ClerkVerifierUnavailable("CLERK_ISSUER_URL is not set")

    try:
        client = _get_jwk_client(jwks_url)
        signing_key = client.get_signing_key_from_jwt(token)
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"require": ["exp", "iss", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Clerk token verification failed: expired")
        return None
    except jwt.InvalidIssuerError:
        logger.warning("Clerk token verification failed: issuer mismatch")
        return None
    except jwt.InvalidSignatureError:
        logger.warning("Clerk token verification failed: bad signature")
        return None
    except (PyJWKClientConnectionError, httpx.HTTPError, json.JSONDecodeError) as e:
        # JSONDecodeError: the JWKS URL answered with something other than
        # JSON (PyJWKClient doesn't wrap it), e.g. a wrong CLERK_JWKS_URL.
        logger.error("Clerk JWKS fetch failed: %s", type(e).__name__)
        raise ClerkVerifierUnavailable("Clerk JWKS fetch failed") from e
    except jwt.PyJWTError as e:
        logger.warning("Clerk token verification failed: %s", type(e).__name__)
        return None

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        logger.warning("Clerk token verification failed: missing sub")
        return None
    return sub
