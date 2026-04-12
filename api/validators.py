"""Shared input validators for API routes."""

import re

# Matches equity tickers: 1-10 uppercase alphanumeric chars, dots, and hyphens.
# Handles standard (AAPL), class-share (BRK-B), and exchange-suffix (BF.B) formats.
# All callers should normalize to uppercase before matching.
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")
