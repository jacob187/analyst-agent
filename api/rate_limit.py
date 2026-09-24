"""Sliding-window rate limiting for WebSocket messages and REST endpoints.

Two limiters live here:

* ``check_rate_limit(ip)`` — per-IP message limit for WebSocket chat (10/min).
* ``check_rest_rate_limit(key, bucket, max_calls, window_seconds)`` — generic
  per-key, per-bucket limiter used by LLM-backed REST endpoints. Use
  ``rate_limit_key(user_id, ip)`` to pick a user-id key for authenticated
  callers and fall back to IP for anonymous ones.

All stores are ``cachetools.TTLCache`` instances so memory stays bounded
under rotating/spoofed IPs or user IDs.

``client_ip(conn)`` resolves the IP those limiters are keyed on. Read it from
here rather than ``conn.client.host`` — see its docstring for why.
"""

import ipaddress
import os
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
# One store per window length, each with ttl == the window it serves. A single
# shared store would evict entries before a longer window closed — silently
# handing the caller a fresh quota — and would let high-traffic short-window
# buckets push long-window ones out of a shared maxsize.
_rest_stores: dict[int, TTLCache] = {}


def _rest_store(window_seconds: int) -> TTLCache:
    store = _rest_stores.get(window_seconds)
    if store is None:
        store = TTLCache(maxsize=10_000, ttl=window_seconds)
        _rest_stores[window_seconds] = store
    return store


def clear_rate_limits() -> None:
    """Drop all limiter state. For tests — not called in production paths."""
    _timestamps.clear()
    for store in _rest_stores.values():
        store.clear()


def rate_limit_key(user_id: str | None, ip: str) -> str:
    """Pick a stable rate-limit key: authenticated user > client IP."""
    return f"u:{user_id}" if user_id else f"ip:{ip}"


# ── Client IP resolution ─────────────────────────────────────────────────────
# How many proxies sit between the caller and us. Railway's edge is one hop;
# raise this if another proxy is added in front, and see client_ip() for what
# happens when it's wrong.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "1"))


def client_ip(conn) -> str:
    """Resolve the client IP to key per-IP limits on.

    Counted from the RIGHT of ``X-Forwarded-For``: the leftmost entries are
    whatever the caller sent (trivially forged to rotate past a quota), while
    the rightmost are appended by the proxies in front of us. Picking the
    ``TRUSTED_PROXY_HOPS``-th from the right lands on the hop our own edge
    wrote. Too low reaches an internal proxy IP and collapses every visitor
    into one bucket; too high reaches back into client-supplied text.

    ``conn`` is a Request or WebSocket — both expose ``.headers``/``.client``.
    """
    forwarded = conn.headers.get("x-forwarded-for")
    if forwarded:
        hops = [p.strip() for p in forwarded.split(",") if p.strip()]
        if hops:
            hop = hops[-min(TRUSTED_PROXY_HOPS, len(hops))]
            try:
                ipaddress.ip_address(hop)
                return hop
            except ValueError:
                # Present but unparseable: share one bucket rather than hand
                # out a fresh quota per garbage value.
                return "unknown"

    return conn.client.host if conn.client else "unknown"


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
    store = _rest_store(window_seconds)
    cache_key = f"{bucket}:{key}"
    recent = [t for t in store.get(cache_key, []) if now - t < window_seconds]

    if len(recent) >= max_calls:
        store[cache_key] = recent
        return False

    recent.append(now)
    store[cache_key] = recent
    return True
