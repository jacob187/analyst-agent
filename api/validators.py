"""Shared input validators for API routes."""

import re

# Matches equity tickers: 1-10 uppercase alphanumeric chars, dots, and hyphens.
# Handles standard (AAPL), class-share (BRK-B), and exchange-suffix (BF.B) formats.
# All callers should normalize to uppercase before matching.
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

# Accepts two user-ID formats during the Clerk transition:
#   1. Legacy UUID v4   — anonymous IDs minted by the frontend pre-Clerk (deprecated).
#   2. Clerk session ID — `user_` followed by Clerk's opaque alphanumeric suffix.
# Both formats coexist until all anonymous IDs have aged out of the database.
USER_ID_RE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|user_[A-Za-z0-9]+)$"
)
