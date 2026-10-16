"""Purge cached narrative filing analyses generated from failed extractions.

edgartools 5.28.4 failed every narrative section extraction, and the Filings
pipeline analysed the resulting error text, caching it per accession. Those
rows can't be reliably told apart from good ones, so delete every narrative
analysis created before the fix shipped; they regenerate on demand. XBRL-based
types (balance, income_stmt, cashflow) were unaffected and are kept.

    uv run python -m api.purge_filing_analyses --created-before "2026-10-11 18:00:00"
    uv run python -m api.purge_filing_analyses --created-before "..." --execute

Dry-run by default. ``--created-before`` is UTC (SQLite CURRENT_TIMESTAMP).
"""

import argparse
import asyncio
from datetime import datetime

from api.db import close_db, count_filing_analyses, delete_filing_analyses, init_db

NARRATIVE_TYPES = [
    "risk_10k", "mda_10k", "risk_10q", "mda_10q",
    "business", "cybersecurity", "legal", "market_risk",
    "earnings", "event",
]


def _utc_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid timestamp {value!r}: {e}") from e


async def run(created_before: str, execute: bool) -> int:
    await init_db()
    try:
        counts = await count_filing_analyses(NARRATIVE_TYPES, created_before)
        total = sum(counts.values())
        for atype in NARRATIVE_TYPES:
            print(f"  {atype:14} {counts.get(atype, 0):>6}")
        print(f"  {'TOTAL':14} {total:>6}  (created before {created_before} UTC)")
        if not execute:
            print("Dry run — re-run with --execute to delete.")
            return 0
        deleted = await delete_filing_analyses(NARRATIVE_TYPES, created_before)
        print(f"Deleted {deleted} rows.")
        return deleted
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--created-before", required=True, type=_utc_timestamp,
                        help="UTC cutoff, e.g. '2026-10-11 18:00:00' (fix deploy time)")
    parser.add_argument("--execute", action="store_true", help="actually delete (default: dry run)")
    args = parser.parse_args()
    asyncio.run(run(args.created_before, args.execute))


if __name__ == "__main__":
    main()
