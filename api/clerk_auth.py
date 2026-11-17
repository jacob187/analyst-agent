"""Clerk JWT verification using Clerk's published JWKS.

Verifies session tokens issued by Clerk by:
  1. Locating the signing key by the token's `kid` in Clerk's JWKS (kept in
     memory per JWKS URL, refreshed every ~10 minutes).
  2. Verifying signature (RS256), `exp`, and `iss` claims.

We use raw JWT verification rather than `clerk-backend-api.authenticate_request`
because that helper requires an `httpx.Request` object, which is awkward for
WebSocket auth (we only have a token string). Direct verification works
uniformly across REST and WebSocket contexts.

Configure via env:
  CLERK_SECRET_KEY  — presence of this enables verification (any non-empty value)
  CLERK_ISSUER_URL  — Clerk Frontend API URL (the tokens' `iss`), e.g.
                      https://your-app.clerk.accounts.dev, or
                      https://<frontend>/__clerk for a proxied instance
  CLERK_JWKS_URL    — optional override; defaults to {issuer}/.well-known/jwks.json
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import jwt
from jwt import PyJWK, PyJWKClient

logger = logging.getLogger(__name__)

_JWKS_TTL_SECONDS = 10 * 60  # refresh the key set this often
# urllib applies it per socket operation. Short, so a stalled JWKS host holds
# one worker thread for seconds, not the PyJWKClient default of 30.
_JWKS_FETCH_TIMEOUT_SECONDS = 5.0
# After a failed fetch, don't fetch again for this long: requests use the keys
# already known, or fail fast, instead of each retrying a down endpoint.
_JWKS_RETRY_AFTER_FAILURE_SECONDS = 15.0
# Minimum gap between successful fetches forced by an unknown `kid`, so tokens
# with made-up key ids can't make every request fetch the JWKS.
_JWKS_UNKNOWN_KID_COOLDOWN_SECONDS = 30.0


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


class ClerkVerifierUnavailable(RuntimeError):
    """The token can't be checked at all: verification is misconfigured or
    Clerk's JWKS is unreachable. Distinct from an invalid/expired token."""


def _fetch_signing_keys(jwks_url: str) -> dict[str, PyJWK]:
    """Fetch the JWKS and return its signing keys by `kid`.

    Every failure here is the verifier's, not the token's: unreachable or
    stalled endpoint, a non-http(s) URL, a body that isn't JSON, or JSON that
    isn't a key set with signing keys. All raise `ClerkVerifierUnavailable`.
    """
    try:
        client = PyJWKClient(
            jwks_url, cache_jwk_set=False, timeout=_JWKS_FETCH_TIMEOUT_SECONDS
        )
        keys = client.get_signing_keys()
    except (jwt.PyJWTError, ValueError, OSError) as e:
        # PyJWTError: connection errors, bad URL scheme, not a key set.
        # ValueError: a body that isn't JSON (PyJWKClient doesn't wrap it).
        # OSError: a socket error while reading the body (also unwrapped).
        raise ClerkVerifierUnavailable(f"Clerk JWKS fetch failed: {type(e).__name__}") from e
    return {k.key_id: k for k in keys if k.key_id}


class _JwksKeys:
    """Signing keys from one JWKS URL, refreshed without blocking verification.

    At most one thread fetches at a time, and no lock is held during the
    fetch. While a refresh is in flight, or after one fails, keys already
    known keep verifying; only an unknown `kid` waits for the in-flight fetch
    (bounded by the fetch timeout) or fails fast.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._cond = threading.Condition()
        self._keys: dict[str, PyJWK] = {}
        self._fetched_at: float | None = None  # last successful fetch (monotonic)
        self._failed_at: float | None = None  # last fetch, if it failed
        self._refreshing = False

    def signing_key(self, kid: str) -> PyJWK | None:
        """The key for `kid`, or None when an up-to-date key set lacks it (a
        bad token). Raises `ClerkVerifierUnavailable` when `kid` is unknown
        and the JWKS can't be fetched right now."""
        waited = False
        with self._cond:
            while True:
                now = time.monotonic()
                key = self._keys.get(kid)
                stale = self._fetched_at is None or now - self._fetched_at >= _JWKS_TTL_SECONDS
                if key is not None and not stale:
                    return key
                if self._refreshing:
                    if key is not None:
                        return key
                    if waited:
                        raise ClerkVerifierUnavailable("Clerk JWKS fetch still in progress")
                    self._cond.wait_for(
                        lambda: not self._refreshing, timeout=_JWKS_FETCH_TIMEOUT_SECONDS
                    )
                    waited = True
                    continue
                if (
                    self._failed_at is not None
                    and now - self._failed_at < _JWKS_RETRY_AFTER_FAILURE_SECONDS
                ):
                    if key is not None:
                        return key
                    raise ClerkVerifierUnavailable("Clerk JWKS is unavailable")
                if (
                    key is None
                    and not stale
                    and self._failed_at is None
                    and now - self._fetched_at < _JWKS_UNKNOWN_KID_COOLDOWN_SECONDS
                ):
                    return None
                self._refreshing = True
                break

        try:
            keys = _fetch_signing_keys(self.url)
        except ClerkVerifierUnavailable as e:
            with self._cond:
                self._failed_at = time.monotonic()
                self._refreshing = False
                self._cond.notify_all()
            logger.error(
                "%s: %s (verifying with %d known key(s) until it recovers)",
                e, e.__cause__, len(self._keys),
            )
            if key is not None:
                return key
            raise
        except BaseException:
            with self._cond:
                self._refreshing = False
                self._cond.notify_all()
            raise

        with self._cond:
            self._keys = keys
            self._fetched_at = time.monotonic()
            self._failed_at = None
            self._refreshing = False
            self._cond.notify_all()
        return keys.get(kid)


_jwks_keys: dict[str, _JwksKeys] = {}
_jwks_keys_lock = threading.Lock()


def _keys_for(jwks_url: str) -> _JwksKeys:
    with _jwks_keys_lock:
        keys = _jwks_keys.get(jwks_url)
        if keys is None:
            keys = _jwks_keys[jwks_url] = _JwksKeys(jwks_url)
        return keys


def _reset_cache_for_tests() -> None:
    with _jwks_keys_lock:
        _jwks_keys.clear()


def verify_clerk_token(token: str) -> str | None:
    """Verify a Clerk session JWT and return the `sub` claim, or None on failure.

    Returns None for a token that fails verification (malformed, expired, bad
    signature, wrong issuer, a key the JWKS doesn't list). Raises
    `ClerkVerifierUnavailable` when the verifier itself can't run
    (CLERK_ISSUER_URL unset, JWKS unreachable or not a usable key set while
    the token's key isn't already known), so callers don't mistake an outage
    for a signed-out user.

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
        kid = jwt.get_unverified_header(token).get("kid")
    except jwt.PyJWTError as e:
        logger.warning("Clerk token verification failed: %s", type(e).__name__)
        return None
    if not isinstance(kid, str) or not kid:
        logger.warning("Clerk token verification failed: no key id")
        return None

    signing_key = _keys_for(jwks_url).signing_key(kid)
    if signing_key is None:
        logger.warning("Clerk token verification failed: unknown signing key")
        return None

    try:
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
    except jwt.PyJWTError as e:
        logger.warning("Clerk token verification failed: %s", type(e).__name__)
        return None

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        logger.warning("Clerk token verification failed: missing sub")
        return None
    return sub
