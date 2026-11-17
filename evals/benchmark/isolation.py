"""Per-run isolation: a fresh SQLite database, cold agent caches and a quieter OpenTelemetry logger.

Tools persist LLM filing analyses through ``api.db``. A benchmark run must not
reuse rows from an earlier run or from the dev app's ``data/chats.db``
(otherwise later runs look artificially cheap), and it must not write there.
Nor may it reuse the agent's process-wide caches an earlier run in the same
process warmed (otherwise later runs look artificially fast).
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

DB_ISOLATED = "isolated"
_DB_FILE = "benchmark.db"
# The SDK logs "String attribute value exceeds max length ..., truncating." from here.
_OTEL_ATTRIBUTES_LOGGER = "opentelemetry.attributes"


def reset_process_caches() -> None:
    """Clear the agent's process-wide caches so a run cannot reuse an earlier run's work.

    ``compare-models`` runs every model in one process. Without this, a later model's run
    would find the SEC retrievers (with their parsed filing sections), latest-filing
    lookups, Tavily research results and Yahoo Finance retrievers an earlier run built,
    and its latency would look better than it is. The per-key asyncio locks go too: they
    belong to the event loop of the run that made them. These are the agent's own module
    caches; caches inside libraries such as edgartools are outside the harness's control.
    """
    from agents.sec_workflow import analysis_cache
    from agents.technical_workflow import indicator_window
    from agents.tools.research_tools import clear_research_cache

    analysis_cache.clear_caches()
    clear_research_cache()
    indicator_window._retriever_cache.clear()


@asynccontextmanager
async def isolated_db() -> AsyncIterator[Path]:
    """Point ``api.db`` at a new temporary SQLite database for the duration, with cold caches.

    ``api.db`` keeps one process-wide connection, so any connection already open
    (e.g. to the default path) is closed first; the new database is created and
    initialised before anything runs, and the agent's process-wide caches are
    cleared (``reset_process_caches``). On exit the connection is closed, the
    previous path restored and the temp directory deleted. Not re-entrant and
    not safe for concurrent runs in one process (the DB layer is a singleton).
    """
    import api.db as db

    reset_process_caches()
    await db.close_db()
    previous_path = db.DB_PATH
    with tempfile.TemporaryDirectory(prefix="benchmark-db-") as tmp:
        path = Path(tmp) / _DB_FILE
        db.DB_PATH = path
        try:
            await db.init_db()
            yield path
        finally:
            await db.close_db()
            db.DB_PATH = previous_path


@contextmanager
def quiet_attribute_truncation() -> Iterator[None]:
    """Silence the OTel "attribute exceeds max length, truncating" warnings.

    The attribute cap is intentional (8192 in the app, ``metrics.SPAN_ATTRIBUTE_CAP``
    in benchmark runs) and filing text in tool outputs can trip it. Errors still get through.
    """
    otel_logger = logging.getLogger(_OTEL_ATTRIBUTES_LOGGER)
    previous = otel_logger.level
    otel_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        otel_logger.setLevel(previous)
