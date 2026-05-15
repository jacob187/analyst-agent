"""Sliding-window rate limiting for WebSocket messages and REST endpoints.

Two limiters live here:

* ``check_rate_limit(ip)`` — per-IP message limit for WebSocket chat (10/min).
* ``check_rest_rate_limit(key, bucket, max_calls, window_seconds)`` — generic
  per-key, per-bucket limiter used by LLM-backed REST endpoints. Use
  ``rate_limit_key(user_id, ip)`` to pick a user-id key for authenticated
  callers and fall back to IP for anonymous ones.

Both stores are ``cachetools.TTLCache`` instances so memory stays bounded
under rotating/spoofed IPs or user IDs.
"""

import time

from cachetools import TTLCache

# ── WebSocket message limiter ────────────────────────────────────────────────
MAX_MESSAGES = 10
WINDOW_SECONDS = 60

_timestamps: TTLCache = TTLCache(maxsize=10_000, ttl=WINDOW_SECONDS)


def check_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if rate limited."""
    now = time.time()
    cutoff = now - WINDOW_SECONDS
    recent = [t for t in _timestamps.get(ip, []) if t > cutoff]

    if len(recent) >= MAX_MESSAGES:
        _timestamps[ip] = recent
        return False

    recent.append(now)
    _timestamps[ip] = recent
    return True


# ── REST endpoint limiter ────────────────────────────────────────────────────
# Largest window we expect any caller to use. TTL eviction protects memory
# even if a key never gets revisited.
_REST_MAX_WINDOW = 3600

_rest_timestamps: TTLCache = TTLCache(maxsize=10_000, ttl=_REST_MAX_WINDOW)


def rate_limit_key(user_id: str | None, ip: str) -> str:
    """Pick a stable rate-limit key: authenticated user > client IP."""
    return f"u:{user_id}" if user_id else f"ip:{ip}"


def check_rest_rate_limit(
    key: str, bucket: str, max_calls: int, window_seconds: int
) -> bool:
    """Sliding-window limiter scoped by bucket.

    ``bucket`` namespaces the limit so different endpoints don't share quota
    for the same user. Returns True if the call is allowed, False if the
    caller has already hit ``max_calls`` within the trailing
    ``window_seconds``.
    """
    now = time.time()
    cache_key = f"{bucket}:{key}"
    recent = [t for t in _rest_timestamps.get(cache_key, []) if now - t < window_seconds]

    if len(recent) >= max_calls:
        _rest_timestamps[cache_key] = recent
        return False

    recent.append(now)
    _rest_timestamps[cache_key] = recent
    return True
