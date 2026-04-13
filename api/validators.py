"""Shared input validators for API routes."""

import re

# Matches equity tickers: 1-10 uppercase alphanumeric chars, dots, and hyphens.
# Handles standard (AAPL), class-share (BRK-B), and exchange-suffix (BF.B) formats.
# All callers should normalize to uppercase before matching.
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

# UUID v4 format — used to validate anonymous user IDs sent by the frontend.
USER_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
