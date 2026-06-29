"""Shared input validators for API routes."""

import re

# Matches equity tickers: 1-10 uppercase alphanumeric chars, dots, and hyphens.
# Handles standard (AAPL), class-share (BRK-B), and exchange-suffix (BF.B) formats.
# All callers should normalize to uppercase before matching.
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

# Accepts two user-ID formats:
#   1. Clerk account ID — `user_` followed by Clerk's opaque alphanumeric suffix.
#      Verified against a Clerk session token when Clerk is configured. This is
#      the only id the frontend sends today; it unlocks DB persistence.
#   2. Legacy UUID v4 — kept for backward-compat only. The frontend no longer
#      mints anonymous UUIDs; anonymous visitors send no id and have no DB data.
USER_ID_RE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|user_[A-Za-z0-9]+)$"
)
