"""Per-IP rate limiting for WebSocket messages.

Uses a sliding window: tracks timestamps of recent messages per IP.
Old timestamps are pruned on each check to prevent memory growth.
"""

import time
from collections import defaultdict

MAX_MESSAGES = 10
WINDOW_SECONDS = 60

_timestamps: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if rate limited."""
    now = time.time()
    cutoff = now - WINDOW_SECONDS
    _timestamps[ip] = [t for t in _timestamps[ip] if t > cutoff]

    if len(_timestamps[ip]) >= MAX_MESSAGES:
        return False

    _timestamps[ip].append(now)
    return True
