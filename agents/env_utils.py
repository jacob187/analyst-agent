"""Lenient numeric settings from the environment.

A malformed value (not a number, not finite, below ``minimum``) falls back to
the default with a warning instead of crashing startup or a request. The
warning is logged once per variable and value, so settings read per call
don't flood the log. Unset or blank means the default, silently.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_N = TypeVar("_N", int, float)

_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()


def _env_number(name: str, default: _N, parse: Callable[[str], _N], minimum: _N | None) -> _N:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = parse(raw.strip())
    except ValueError:
        value = None
    # Only floats can be non-finite; math.isfinite would overflow on a huge int.
    non_finite = isinstance(value, float) and not math.isfinite(value)
    if value is None or non_finite or (minimum is not None and value < minimum):
        with _warned_lock:
            first = (name, raw) not in _warned
            _warned.add((name, raw))
        if first:
            logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    return value


def env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    """``float(os.environ[name])``, or ``default`` when unset or invalid."""
    return _env_number(name, default, float, minimum)


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """``int(os.environ[name])``, or ``default`` when unset or invalid."""
    return _env_number(name, default, int, minimum)
