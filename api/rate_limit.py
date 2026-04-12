"""Per-IP rate limiting for WebSocket messages.

Uses a sliding window: tracks timestamps of recent messages per IP.
Stale timestamps are pruned on each check. IPs with no recent traffic are
evicted to prevent unbounded memory growth under rotating/spoofed IPs.
"""

import time

MAX_MESSAGES = 10
WINDOW_SECONDS = 60

_timestamps: dict[str, list[float]] = {}


def check_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if rate limited."""
    now = time.time()
    cutoff = now - WINDOW_SECONDS
    recent = [t for t in _timestamps.get(ip, []) if t > cutoff]

    # Evict the key when all previous timestamps have expired.
    # The current request recreates the entry below.
    if not recent:
        _timestamps.pop(ip, None)

    if len(recent) >= MAX_MESSAGES:
        _timestamps[ip] = recent  # persist pruned list so next check is fast
        return False

    recent.append(now)
    _timestamps[ip] = recent
    return True
