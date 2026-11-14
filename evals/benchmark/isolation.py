"""Per-run isolation: a fresh SQLite database and a quieter OpenTelemetry logger.

Tools persist LLM filing analyses through ``api.db``. A benchmark run must not
reuse rows from an earlier run or from the dev app's ``data/chats.db``
(otherwise later runs look artificially cheap), and it must not write there.
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


@asynccontextmanager
async def isolated_db() -> AsyncIterator[Path]:
    """Point ``api.db`` at a new temporary SQLite database for the duration.

    ``api.db`` keeps one process-wide connection, so any connection already open
    (e.g. to the default path) is closed first; the new database is created and
    initialised before anything runs. On exit the connection is closed, the
    previous path restored and the temp directory deleted. Not re-entrant and
    not safe for concurrent runs in one process (the DB layer is a singleton).
    """
    import api.db as db

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

    The 8192-character attribute cap is intentional (``api/tracing.py``) and
    filing text in tool outputs trips it constantly. Errors still get through.
    """
    otel_logger = logging.getLogger(_OTEL_ATTRIBUTES_LOGGER)
    previous = otel_logger.level
    otel_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        otel_logger.setLevel(previous)
