"""Shared SEC filing-analysis cache for the company filings endpoint
(``api/routes/company.py``) and the chat SEC tools (``agents/tools/sec_tools.py``).

- ``get_retriever(ticker)`` — the process-wide per-ticker ``SECDataRetrieval``.
- ``latest_filing(ticker, form)`` — the latest filing per (ticker, form), asked
  of SEC EDGAR at most once per ``FILING_FRESHNESS_TTL_SECONDS`` (default 4h).
  A lookup that finds a newer filing also points the shared retriever at it,
  so section text extracted afterwards comes from the filing the ref names.
- ``get_or_analyze(ticker, filing, analysis_type, generate)`` — read-through
  cache over the ``filing_analyses`` table keyed (ticker, form, accession,
  analysis_type): a hit costs no LLM call; concurrent identical misses
  generate once.
- ``run_analysis(llm, ticker, analysis_type, raw)`` — the single Pydantic →
  dict path, so a stored analysis has the same shape whichever caller made it.

Every served result carries its filing reference (filing date, accession,
analysed-at) so answers can say which filing they describe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import weakref
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from cachetools import LRUCache, TTLCache

logger = logging.getLogger(__name__)

# Filings land a few times a quarter (8-Ks: any day). Four hours keeps a
# just-published filing from being missed for long while capping EDGAR
# submissions requests at ~6 per (ticker, form) per day.
DEFAULT_FRESHNESS_TTL_SECONDS = 4 * 3600

FORMS = ("10-K", "10-Q", "8-K", "20-F")

_MISS = object()

# Guards the two caches below: worker threads (edgartools calls) and the
# event loop both touch them.
_lock = threading.Lock()
_shared_retrievers: LRUCache = LRUCache(maxsize=128)
# (TICKER, form) -> (FilingRef | None, looked_up_at monotonic seconds,
#                    weakref to the retriever the lookup refreshed)
_latest_filings: LRUCache = LRUCache(maxsize=1024)

# Per-key asyncio locks (event loop only). TTL eviction bounds growth; a
# request already holding an evicted lock keeps it alive.
_latest_locks: TTLCache = TTLCache(maxsize=4096, ttl=600)
_analysis_locks: TTLCache = TTLCache(maxsize=4096, ttl=600)

_clock = time.monotonic


@dataclass(frozen=True)
class FilingRef:
    """A filing as the ``filing_analyses`` key sees it. ``form`` is the
    key's form_type (the filings endpoint stores 20-F analyses under 10-K)."""

    form: str
    accession: str
    filing_date: str


@dataclass(frozen=True)
class CachedAnalysis:
    analysis: dict[str, Any]
    filing_date: str
    accession: str
    analyzed_at: str
    generated: bool

    def filing_reference(self) -> dict[str, str]:
        return {
            "filing_date": self.filing_date,
            "accession": self.accession,
            "analyzed_at": self.analyzed_at,
        }

    def served(self) -> dict[str, Any]:
        """The analysis as served to clients: stored fields plus its filing
        reference (never part of the stored JSON)."""
        return {**self.analysis, "filing_reference": self.filing_reference()}


def freshness_ttl_seconds() -> float:
    """``FILING_FRESHNESS_TTL_SECONDS`` (read per call), else the default."""
    raw = os.getenv("FILING_FRESHNESS_TTL_SECONDS")
    if raw is None:
        return DEFAULT_FRESHNESS_TTL_SECONDS
    try:
        ttl = float(raw)
    except ValueError:
        ttl = -1.0
    if ttl < 0:
        logger.warning(
            "Ignoring invalid FILING_FRESHNESS_TTL_SECONDS=%r; using %ss",
            raw, DEFAULT_FRESHNESS_TTL_SECONDS,
        )
        return DEFAULT_FRESHNESS_TTL_SECONDS
    return ttl


def get_retriever(ticker: str):
    """Process-wide ``SECDataRetrieval`` for ``ticker``. Blocking (edgartools):
    call from a worker thread."""
    from agents.sec_workflow import get_SEC_data

    key = ticker.upper()
    with _lock:
        retriever = _shared_retrievers.get(key)
    if retriever is not None:
        return retriever
    retriever = get_SEC_data.SECDataRetrieval(key)
    with _lock:
        existing = _shared_retrievers.get(key)
        if existing is not None:
            return existing
        _shared_retrievers[key] = retriever
    return retriever


def _lock_for(store: TTLCache, key: tuple) -> asyncio.Lock:
    lock = store.get(key)
    if lock is None:
        lock = asyncio.Lock()
        store[key] = lock
    return lock


def _fresh_ref(key: tuple[str, str]) -> Any:
    """The cached ref, or _MISS once it is older than the TTL — or was looked
    up through a retriever that has since been evicted: its replacement loads
    whatever is latest *now*, so an older ref would key that filing's
    analysis under the wrong accession."""
    with _lock:
        entry = _latest_filings.get(key)
        current = _shared_retrievers.get(key[0])
    if entry is None:
        return _MISS
    ref, looked_up_at, looked_up_with = entry
    if looked_up_with() is not current or current is None:
        return _MISS
    if _clock() - looked_up_at >= freshness_ttl_seconds():
        return _MISS
    return ref


def _refresh(ticker: str, form: str):
    retriever = get_retriever(ticker)
    return retriever, retriever.refresh_latest_filing(form)


async def latest_filing(ticker: str, form: str) -> FilingRef | None:
    """Latest ``form`` filing for ``ticker`` (None if it has none).

    Asks SEC EDGAR only when the (ticker, form) entry is older than the
    freshness TTL; concurrent callers share one lookup. Lookup failures raise
    and are not cached.
    """
    if form not in FORMS:
        raise ValueError(f"Unsupported form: {form}")
    key = (ticker.upper(), form)
    ref = _fresh_ref(key)
    if ref is not _MISS:
        return ref
    async with _lock_for(_latest_locks, key):
        ref = _fresh_ref(key)
        if ref is not _MISS:
            return ref
        retriever, meta = await asyncio.to_thread(_refresh, key[0], form)
        ref = None if meta is None else FilingRef(form, meta.accession, meta.filing_date)
        with _lock:
            _latest_filings[key] = (ref, _clock(), weakref.ref(retriever))
        return ref


def _iso_utc(created_at: str | None) -> str:
    """SQLite CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS', UTC) → ISO 8601."""
    if not created_at:
        return ""
    try:
        parsed = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(created_at)
    return parsed.replace(tzinfo=timezone.utc).isoformat()


async def _read(key: tuple[str, str, str, str], filing: FilingRef) -> CachedAnalysis | None:
    from api import db

    try:
        row = await db.get_filing_analysis(*key)
    except Exception as e:
        # No usable DB (e.g. never initialised): behave as a miss so callers
        # still get an analysis, just uncached.
        logger.warning("filing_analyses read failed for %s: %s", key, e)
        return None
    if not row:
        return None
    return CachedAnalysis(
        analysis=json.loads(row["analysis_json"]),
        filing_date=filing.filing_date,
        accession=filing.accession,
        analyzed_at=_iso_utc(row.get("created_at")),
        generated=False,
    )


async def get_or_analyze(
    ticker: str,
    filing: FilingRef,
    analysis_type: str,
    generate: Callable[[], Awaitable[dict[str, Any]]] | None = None,
) -> CachedAnalysis | None:
    """The cached analysis for (ticker, filing, analysis_type), generating
    and saving it on a miss.

    ``generate`` returns the analysis dict (normally via ``run_analysis``).
    With ``generate=None`` a miss returns None (cache-only / keyless callers).
    Identical concurrent misses generate once: later callers wait and read
    the saved row. A failing ``generate`` raises and nothing is saved.
    """
    from api import db

    key = (ticker.upper(), filing.form, filing.accession, analysis_type)
    hit = await _read(key, filing)
    if hit is not None or generate is None:
        return hit
    async with _lock_for(_analysis_locks, key):
        hit = await _read(key, filing)
        if hit is not None:
            return hit
        analysis = await generate()
        try:
            await db.save_filing_analysis(*key, json.dumps(analysis))
        except Exception as e:
            logger.warning("filing_analyses save failed for %s: %s", key, e)
        saved = await _read(key, filing)
        if saved is not None:
            return replace(saved, generated=True)
        return CachedAnalysis(
            analysis=analysis,
            filing_date=filing.filing_date,
            accession=filing.accession,
            analyzed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            generated=True,
        )


async def run_analysis(
    llm, ticker: str, analysis_type: str, raw_data: dict[str, Any],
) -> dict[str, Any]:
    """Run one LLM analysis (``llm.ainvoke``, so the caller's callbacks apply)
    and return the Pydantic model as the dict that gets stored."""
    from agents.sec_workflow import sec_llm_models

    processor = sec_llm_models.SECDocumentProcessor(llm)
    result = await processor.aanalyze(analysis_type, ticker, raw_data)
    return result.model_dump()


def eightk_input(retriever, overview: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(kind, raw)`` LLM input for the latest 8-K: ``"earnings"`` (Item 2.02
    with a parseable press release) or ``"event"`` (primary item text, empty
    when it failed to extract). Blocking (edgartools)."""
    if overview.get("has_earnings"):
        return "earnings", {**retriever.get_earnings_data(), "metadata": metadata}
    items = overview.get("items", [])
    primary_item = items[0] if items else None
    item_text = ""
    if primary_item:
        item_result = retriever.get_8k_item(primary_item)
        item_text = item_result.get("text", "") if item_result.get("found") else ""
    return "event", {
        "content_type": overview.get("content_type", "other"),
        "items": items,
        "context": overview.get("context", ""),
        "text": item_text,
        "metadata": metadata,
    }
