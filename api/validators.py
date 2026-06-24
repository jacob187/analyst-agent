"""Shared input validators for API routes."""

import re

# Matches equity tickers: 1-10 uppercase alphanumeric chars, dots, and hyphens.
# Handles standard (AAPL), class-share (BRK-B), and exchange-suffix (BF.B) formats.
# All callers should normalize to uppercase before matching.
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

# Accepts two user-ID formats (progressive auth):
#   1. Anonymous UUID v4 — per-browser ID minted by the frontend for signed-out
#      visitors. Trusted as-is; gives data isolation without a Clerk account.
#   2. Clerk account ID  — `user_` followed by Clerk's opaque alphanumeric suffix.
#      Verified against a Clerk session token when Clerk is configured.
USER_ID_RE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|user_[A-Za-z0-9]+)$"
)
