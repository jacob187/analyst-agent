import aiosqlite
import os
import uuid
from pathlib import Path

# Configurable via env var for deployment (e.g., DB_PATH=/data/analyst.db on Fly/Koyeb).
# Defaults to project root/data/chats.db for local development.
DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).parent.parent / "data" / "chats.db"))

# Persistent connection
_db: aiosqlite.Connection | None = None

async def get_db() -> aiosqlite.Connection:
    """Get or create the persistent database connection."""
    global _db
    if _db is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(DB_PATH)
        _db.row_factory = aiosqlite.Row
        # WAL mode allows concurrent reads during writes and prevents readers
        # from blocking writers. busy_timeout retries instead of failing
        # immediately when a write lock is held.
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA busy_timeout=5000")
    return _db

async def init_db():
    """Initialize database and create tables if they don't exist."""
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            summary TEXT,
            user_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration: add summary column to existing databases that predate this column
    try:
        await db.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
        await db.commit()
    except Exception:
        pass  # Column already exists
    # Migration: add model column for multi-model support
    try:
        await db.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
        await db.commit()
    except Exception:
        pass  # Column already exists
    # Migration: add user_id column for per-user isolation
    try:
        await db.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
        await db.commit()
    except Exception:
        pass  # Column already exists
    await db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_created
        ON sessions(created_at DESC)
    """)
    # NOTE: settings table removed — API keys now live in browser localStorage
    # and are sent per-request (WebSocket auth message or request headers).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, ticker)
        )
    """)

    # --- Central company entity (all other tables FK to this) ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- Filing metadata cache (content lives on disk or in pgvector later) ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS filings_cache (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            period_of_report TEXT,
            accession_number TEXT,
            filing_url TEXT,
            downloaded_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES companies(ticker)
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_filings_cache_ticker
        ON filings_cache(ticker)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_filings_cache_form
        ON filings_cache(ticker, form_type)
    """)

    # --- Briefing persistence ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS briefings (
            id TEXT PRIMARY KEY,
            market_regime TEXT NOT NULL,
            market_positioning TEXT NOT NULL,
            alerts TEXT NOT NULL,
            thinking TEXT,
            raw_json TEXT NOT NULL,
            user_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_briefings_created
        ON briefings(created_at)
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS briefing_tickers (
            id TEXT PRIMARY KEY,
            briefing_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            change_pct REAL NOT NULL,
            technical_signal TEXT NOT NULL,
            news_summary TEXT NOT NULL,
            news_url TEXT,
            outlook TEXT NOT NULL,
            FOREIGN KEY (briefing_id) REFERENCES briefings(id),
            FOREIGN KEY (ticker) REFERENCES companies(ticker)
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_briefing_tickers_ticker
        ON briefing_tickers(ticker)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_briefing_tickers_briefing
        ON briefing_tickers(briefing_id)
    """)

    # --- Filing analysis persistence (LLM-generated summaries, cached per accession) ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS filing_analyses (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            form_type TEXT NOT NULL,
            accession_number TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES companies(ticker),
            UNIQUE(ticker, form_type, accession_number, analysis_type)
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_filing_analyses_lookup
        ON filing_analyses(ticker, form_type, accession_number)
    """)

    # Migration: add user_id to briefings for per-user isolation
    try:
        await db.execute("ALTER TABLE briefings ADD COLUMN user_id TEXT")
        await db.commit()
    except Exception:
        pass  # Column already exists

    # Migration: watchlist PK change from (ticker) to (user_id, ticker).
    # SQLite doesn't support ALTER TABLE to change a PK, so we recreate.
    # Only runs if the old schema (ticker-only PK) is detected.
    try:
        async with db.execute("PRAGMA table_info(watchlist)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        if "user_id" not in columns:
            await db.execute("""
                CREATE TABLE watchlist_v2 (
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, ticker)
                )
            """)
            # Old rows have no user_id — they'll be orphaned (intentional).
            await db.execute("DROP TABLE watchlist")
            await db.execute("ALTER TABLE watchlist_v2 RENAME TO watchlist")
            await db.commit()
    except Exception:
        pass  # Migration already applied or table doesn't exist yet

    # Indexes for user-scoped queries
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_briefings_user ON briefings(user_id)
    """)

    # Migration: backfill companies from existing watchlist rows
    await db.execute("""
        INSERT OR IGNORE INTO companies (ticker, added_at)
        SELECT ticker, added_at FROM watchlist
    """)

    await db.commit()

_orphans_claimed = False

async def claim_orphaned_data(user_id: str) -> None:
    """Assign any rows with NULL/empty user_id to the given user.

    Runs at most once per process — after the first call it's a no-op.
    This handles the migration from pre-user-isolation databases where
    existing sessions and briefings have no user_id.
    """
    global _orphans_claimed
    if _orphans_claimed:
        return
    _orphans_claimed = True

    db = await get_db()
    await db.execute(
        "UPDATE sessions SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
        (user_id,),
    )
    await db.execute(
        "UPDATE briefings SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
        (user_id,),
    )
    await db.commit()


async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None

async def create_session(ticker: str, user_id: str, model: str | None = None) -> str:
    """Create a new chat session. Returns session ID.

    Also ensures a companies row exists for the ticker.
    """
    session_id = str(uuid.uuid4())
    db = await get_db()
    ticker = ticker.upper()
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.execute(
        "INSERT INTO sessions (id, ticker, user_id, model) VALUES (?, ?, ?, ?)",
        (session_id, ticker, user_id, model)
    )
    await db.commit()
    return session_id

async def get_or_create_session(ticker: str, user_id: str, model: str | None = None) -> str:
    """
    Return the existing session ID for this ticker+user, or create one.

    One session per ticker per user is enforced at the application level.
    Entering AAPL always resumes that user's AAPL conversation — delete
    the session from History to start fresh.
    """
    db = await get_db()
    ticker = ticker.upper()
    async with db.execute(
        "SELECT id FROM sessions WHERE ticker = ? AND user_id = ?", (ticker, user_id)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return row["id"]
    # No existing session — create one (and ensure company exists)
    session_id = str(uuid.uuid4())
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.execute(
        "INSERT INTO sessions (id, ticker, user_id, model) VALUES (?, ?, ?, ?)",
        (session_id, ticker, user_id, model),
    )
    await db.commit()
    return session_id

async def get_session_by_ticker(ticker: str, user_id: str) -> dict | None:
    """Return session metadata for a ticker+user, or None if no session exists."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, user_id, model, created_at FROM sessions WHERE ticker = ? AND user_id = ?",
        (ticker.upper(), user_id)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return {"id": row["id"], "ticker": row["ticker"], "user_id": row["user_id"], "model": row["model"], "created_at": row["created_at"]}
        return None

async def update_session_summary(session_id: str, summary: str) -> None:
    """Store a rolling summary of the conversation for context compression."""
    db = await get_db()
    await db.execute(
        "UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id)
    )
    await db.commit()

async def save_message(session_id: str, role: str, content: str) -> str:
    """Save a message to the database. Returns message ID."""
    message_id = str(uuid.uuid4())
    db = await get_db()
    await db.execute(
        "INSERT INTO messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
        (message_id, session_id, role, content)
    )
    await db.commit()
    return message_id

async def get_session_messages(session_id: str) -> list[dict]:
    """Get all messages for a session."""
    db = await get_db()
    async with db.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at",
        (session_id,)
    ) as cursor:
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"], "created_at": row["created_at"]} for row in rows]

async def get_sessions(user_id: str, limit: int = 50) -> list[dict]:
    """Get recent sessions for a user."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, model, created_at FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ) as cursor:
        rows = await cursor.fetchall()
        return [{"id": row["id"], "ticker": row["ticker"], "model": row["model"], "created_at": row["created_at"]} for row in rows]

async def get_tickers(user_id: str, limit: int = 50) -> list[dict]:
    """Return companies this user has interacted with (sessions or watchlist).

    Only shows companies where the user has at least one session or a
    watchlist entry — not every company in the global table.
    """
    db = await get_db()
    async with db.execute(
        """
        SELECT c.ticker,
               COUNT(s.id)                            AS session_count,
               COALESCE(MAX(s.created_at), c.added_at) AS last_active
        FROM companies c
        LEFT JOIN sessions s ON s.ticker = c.ticker AND s.user_id = ?
        WHERE c.ticker IN (
            SELECT ticker FROM sessions WHERE user_id = ?
            UNION
            SELECT ticker FROM watchlist WHERE user_id = ?
        )
        GROUP BY c.ticker
        ORDER BY last_active DESC
        LIMIT ?
        """,
        (user_id, user_id, user_id, limit)
    ) as cursor:
        rows = await cursor.fetchall()
        return [
            {"ticker": row["ticker"], "session_count": row["session_count"], "last_active": row["last_active"]}
            for row in rows
        ]

async def get_sessions_for_ticker(ticker: str, user_id: str, limit: int = 50) -> list[dict]:
    """Return all sessions (with IDs) for a specific ticker and user."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, model, created_at FROM sessions WHERE ticker = ? AND user_id = ? ORDER BY created_at DESC LIMIT ?",
        (ticker.upper(), user_id, limit)
    ) as cursor:
        rows = await cursor.fetchall()
        return [{"id": row["id"], "ticker": row["ticker"], "model": row["model"], "created_at": row["created_at"]} for row in rows]

async def get_session(session_id: str) -> dict | None:
    """Get a specific session by ID, including any stored summary and model."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, summary, user_id, model, created_at FROM sessions WHERE id = ?",
        (session_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "ticker": row["ticker"],
                "summary": row["summary"],
                "user_id": row["user_id"],
                "model": row["model"],
                "created_at": row["created_at"],
            }
        return None


async def get_watchlist(user_id: str) -> list[dict]:
    """Return watchlist tickers for a user, ordered by added_at."""
    db = await get_db()
    async with db.execute(
        "SELECT ticker, added_at FROM watchlist WHERE user_id = ? ORDER BY added_at",
        (user_id,)
    ) as cursor:
        rows = await cursor.fetchall()
        return [{"ticker": row["ticker"], "added_at": row["added_at"]} for row in rows]


async def get_watchlist_enriched(user_id: str) -> list[dict]:
    """Return watchlist tickers for a user, enriched with company name/sector."""
    db = await get_db()
    async with db.execute("""
        SELECT w.ticker, w.added_at, c.name, c.sector
        FROM watchlist w
        LEFT JOIN companies c ON w.ticker = c.ticker
        WHERE w.user_id = ?
        ORDER BY w.added_at
    """, (user_id,)) as cursor:
        rows = await cursor.fetchall()
        return [
            {"ticker": row["ticker"], "added_at": row["added_at"],
             "name": row["name"], "sector": row["sector"]}
            for row in rows
        ]


async def add_to_watchlist(ticker: str, user_id: str) -> bool:
    """Add ticker for a user. Returns False if already present or limit (10) reached.

    Also ensures a companies row exists for the ticker (upsert).
    """
    db = await get_db()
    ticker = ticker.upper()
    # Check per-user limit
    async with db.execute(
        "SELECT COUNT(*) as cnt FROM watchlist WHERE user_id = ?", (user_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if row["cnt"] >= 10:
            return False
    # Insert (ignore duplicate — PK is (user_id, ticker))
    cursor = await db.execute(
        "INSERT OR IGNORE INTO watchlist (user_id, ticker) VALUES (?, ?)",
        (user_id, ticker)
    )
    # Ensure company record exists
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.commit()
    return cursor.rowcount > 0


async def remove_from_watchlist(ticker: str, user_id: str) -> bool:
    """Remove ticker from a user's watchlist. Returns True if deleted."""
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM watchlist WHERE ticker = ? AND user_id = ?",
        (ticker.upper(), user_id)
    )
    await db.commit()
    return cursor.rowcount > 0


async def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages. Returns True if deleted."""
    db = await get_db()
    # Delete messages first (foreign key constraint)
    await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    # Delete the session
    cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    await db.commit()
    return cursor.rowcount > 0


# =============================================================================
# Companies
# =============================================================================

async def ensure_company(ticker: str) -> None:
    """Ensure a companies row exists for the ticker (idempotent upsert)."""
    db = await get_db()
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker.upper(),))
    await db.commit()


async def get_company(ticker: str) -> dict | None:
    """Return company metadata, or None if not tracked."""
    db = await get_db()
    async with db.execute(
        "SELECT ticker, name, sector, added_at FROM companies WHERE ticker = ?",
        (ticker.upper(),)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return {
                "ticker": row["ticker"],
                "name": row["name"],
                "sector": row["sector"],
                "added_at": row["added_at"],
            }
        return None


async def get_companies() -> list[dict]:
    """Return all tracked companies."""
    db = await get_db()
    async with db.execute(
        "SELECT ticker, name, sector, added_at FROM companies ORDER BY added_at"
    ) as cursor:
        rows = await cursor.fetchall()
        return [
            {"ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "added_at": r["added_at"]}
            for r in rows
        ]


async def get_company_activity(ticker: str, user_id: str) -> dict:
    """Return combined activity timeline for a company, scoped to a user."""
    db = await get_db()
    ticker = ticker.upper()

    async with db.execute(
        "SELECT id, created_at FROM sessions WHERE ticker = ? AND user_id = ? ORDER BY created_at DESC",
        (ticker, user_id)
    ) as cursor:
        sessions = [{"id": r["id"], "created_at": r["created_at"]} for r in await cursor.fetchall()]

    async with db.execute("""
        SELECT b.id, b.market_regime, b.created_at, bt.outlook
        FROM briefing_tickers bt
        JOIN briefings b ON bt.briefing_id = b.id
        WHERE bt.ticker = ? AND b.user_id = ?
        ORDER BY b.created_at DESC
    """, (ticker, user_id)) as cursor:
        briefings = [
            {"id": r["id"], "market_regime": r["market_regime"],
             "created_at": r["created_at"], "outlook": r["outlook"]}
            for r in await cursor.fetchall()
        ]

    return {"ticker": ticker, "sessions": sessions, "briefings": briefings}


async def update_company(ticker: str, name: str | None = None, sector: str | None = None) -> bool:
    """Update company enrichment data. Returns True if row was updated."""
    db = await get_db()
    ticker = ticker.upper()
    sets = []
    params = []
    if name is not None:
        sets.append("name = ?")
        params.append(name)
    if sector is not None:
        sets.append("sector = ?")
        params.append(sector)
    if not sets:
        return False
    params.append(ticker)
    cursor = await db.execute(
        f"UPDATE companies SET {', '.join(sets)} WHERE ticker = ?", params
    )
    await db.commit()
    return cursor.rowcount > 0


# =============================================================================
# Briefings
# =============================================================================

async def save_briefing(raw_json: str, market_regime: str, market_positioning: str,
                        alerts_json: str, thinking: str | None,
                        tickers: list[dict], user_id: str | None = None) -> str:
    """Persist a briefing and its per-ticker data. Returns briefing ID.

    Args:
        raw_json: Full DailyBriefingAnalysis.model_dump_json()
        market_regime: Regime summary string
        market_positioning: Positioning summary string
        alerts_json: JSON-serialized list of alert strings
        thinking: LLM chain-of-thought (may be empty)
        tickers: List of dicts with keys: ticker, price, change_pct,
                 technical_signal, news_summary, news_url, outlook
        user_id: Anonymous user ID (scopes briefing to a user)
    """
    briefing_id = str(uuid.uuid4())
    db = await get_db()
    await db.execute("""
        INSERT INTO briefings (id, market_regime, market_positioning, alerts, thinking, raw_json, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (briefing_id, market_regime, market_positioning, alerts_json, thinking, raw_json, user_id))

    # Batch-insert companies and briefing_tickers in two round-trips instead of
    # 2×N individual execute() calls — executemany sends all rows in one pass.
    await db.executemany(
        "INSERT OR IGNORE INTO companies (ticker) VALUES (?)",
        [(t["ticker"],) for t in tickers],
    )
    await db.executemany(
        "INSERT INTO briefing_tickers "
        "(id, briefing_id, ticker, price, change_pct, technical_signal, "
        "news_summary, news_url, outlook) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                str(uuid.uuid4()), briefing_id, t["ticker"], t["price"],
                t["change_pct"], t["technical_signal"], t["news_summary"],
                t.get("news_url"), t["outlook"],
            )
            for t in tickers
        ],
    )

    await db.commit()
    return briefing_id


async def get_recent_briefings(user_id: str | None = None, limit: int = 10) -> list[dict]:
    """Return most recent briefings with their tickers, scoped to a user.

    A single JOIN replaces the previous N+1 pattern (1 briefings query +
    1 ticker sub-query per row). Python groups the flat rows by briefing id.

    If user_id is None (e.g. called from agent tools without user context),
    returns an empty list to prevent cross-user data leakage.
    """
    if user_id is None:
        return []
    db = await get_db()
    async with db.execute(
        """
        SELECT b.id, b.market_regime, b.market_positioning, b.alerts,
               b.thinking, b.created_at,
               bt.ticker, bt.price, bt.change_pct, bt.technical_signal,
               bt.news_summary, bt.news_url, bt.outlook
        FROM briefings b
        LEFT JOIN briefing_tickers bt ON bt.briefing_id = b.id
        WHERE b.id IN (
            SELECT id FROM briefings WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
        )
        ORDER BY b.created_at DESC
        """,
        (user_id, limit),
    ) as cursor:
        rows = await cursor.fetchall()

    # Group flat rows into nested briefing dicts, preserving order.
    briefings: dict[str, dict] = {}
    for row in rows:
        bid = row["id"]
        if bid not in briefings:
            briefings[bid] = {
                "id": bid,
                "market_regime": row["market_regime"],
                "market_positioning": row["market_positioning"],
                "alerts": row["alerts"],
                "thinking": row["thinking"],
                "created_at": row["created_at"],
                "tickers": [],
            }
        if row["ticker"]:  # LEFT JOIN produces NULL ticker on briefings with no tickers
            briefings[bid]["tickers"].append({
                "ticker": row["ticker"],
                "price": row["price"],
                "change_pct": row["change_pct"],
                "technical_signal": row["technical_signal"],
                "news_summary": row["news_summary"],
                "news_url": row["news_url"],
                "outlook": row["outlook"],
            })
    return list(briefings.values())


async def get_briefing_history(ticker: str, user_id: str, days: int = 30) -> list[dict]:
    """Return briefing entries for a specific ticker+user within the last N days."""
    db = await get_db()
    ticker = ticker.upper()
    async with db.execute("""
        SELECT b.id, b.market_regime, b.created_at,
               bt.price, bt.change_pct, bt.technical_signal,
               bt.news_summary, bt.outlook
        FROM briefing_tickers bt
        JOIN briefings b ON bt.briefing_id = b.id
        WHERE bt.ticker = ?
          AND b.user_id = ?
          AND b.created_at >= datetime('now', ? || ' days')
        ORDER BY b.created_at DESC
    """, (ticker, user_id, f"-{days}")) as cursor:
        rows = await cursor.fetchall()
        return [
            {
                "briefing_id": r["id"], "market_regime": r["market_regime"],
                "created_at": r["created_at"], "price": r["price"],
                "change_pct": r["change_pct"], "technical_signal": r["technical_signal"],
                "news_summary": r["news_summary"], "outlook": r["outlook"],
            }
            for r in rows
        ]


# =============================================================================
# Filings Cache
# =============================================================================

async def save_filing_metadata(ticker: str, form_type: str, filing_date: str,
                               period_of_report: str | None = None,
                               accession_number: str | None = None,
                               filing_url: str | None = None) -> str:
    """Cache SEC filing metadata. Returns filing cache ID.

    The filing_id (returned) can be used as a FK from a future pgvector
    embeddings table to link vector chunks back to their source filing.
    """
    filing_id = str(uuid.uuid4())
    db = await get_db()
    ticker = ticker.upper()
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.execute("""
        INSERT INTO filings_cache
            (id, ticker, form_type, filing_date, period_of_report,
             accession_number, filing_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (filing_id, ticker, form_type, filing_date, period_of_report,
          accession_number, filing_url))
    await db.commit()
    return filing_id


async def get_filing_metadata(ticker: str, form_type: str | None = None) -> list[dict]:
    """Return cached filing metadata for a ticker, optionally filtered by form type."""
    db = await get_db()
    ticker = ticker.upper()
    if form_type:
        query = ("SELECT * FROM filings_cache WHERE ticker = ? AND form_type = ? "
                 "ORDER BY filing_date DESC")
        params = (ticker, form_type)
    else:
        query = "SELECT * FROM filings_cache WHERE ticker = ? ORDER BY filing_date DESC"
        params = (ticker,)

    async with db.execute(query, params) as cursor:
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"], "ticker": r["ticker"], "form_type": r["form_type"],
                "filing_date": r["filing_date"], "period_of_report": r["period_of_report"],
                "accession_number": r["accession_number"], "filing_url": r["filing_url"],
                "downloaded_at": r["downloaded_at"], "created_at": r["created_at"],
            }
            for r in rows
        ]


async def mark_filing_downloaded(filing_id: str) -> bool:
    """Mark a filing as downloaded (sets downloaded_at timestamp)."""
    db = await get_db()
    cursor = await db.execute(
        "UPDATE filings_cache SET downloaded_at = CURRENT_TIMESTAMP WHERE id = ?",
        (filing_id,)
    )
    await db.commit()
    return cursor.rowcount > 0


# =============================================================================
# Filing Analyses (LLM-generated summaries, cached per accession number)
# =============================================================================

async def save_filing_analysis(ticker: str, form_type: str, accession_number: str,
                               analysis_type: str, analysis_json: str) -> str:
    """Persist an LLM filing analysis. Returns analysis ID.

    Uses INSERT OR REPLACE keyed on the UNIQUE(ticker, form_type,
    accession_number, analysis_type) constraint. If the same analysis
    is re-run for the same filing, it overwrites the previous result.
    """
    analysis_id = str(uuid.uuid4())
    db = await get_db()
    ticker = ticker.upper()
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.execute("""
        INSERT OR REPLACE INTO filing_analyses
            (id, ticker, form_type, accession_number, analysis_type, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (analysis_id, ticker, form_type, accession_number, analysis_type, analysis_json))
    await db.commit()
    return analysis_id


async def get_filing_analysis(ticker: str, form_type: str,
                              accession_number: str,
                              analysis_type: str) -> dict | None:
    """Return a cached filing analysis, or None on cache miss.

    Cache miss happens when:
    - The analysis was never run for this filing
    - A new filing was published (different accession number)
    """
    db = await get_db()
    async with db.execute("""
        SELECT id, analysis_json, created_at
        FROM filing_analyses
        WHERE ticker = ? AND form_type = ? AND accession_number = ? AND analysis_type = ?
    """, (ticker.upper(), form_type, accession_number, analysis_type)) as cursor:
        row = await cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "analysis_json": row["analysis_json"],
                "created_at": row["created_at"],
            }
        return None


async def get_all_filing_analyses(ticker: str, accession_number: str) -> list[dict]:
    """Return all cached analyses for a specific filing (by accession number)."""
    db = await get_db()
    async with db.execute("""
        SELECT id, form_type, analysis_type, analysis_json, created_at
        FROM filing_analyses
        WHERE ticker = ? AND accession_number = ?
        ORDER BY analysis_type
    """, (ticker.upper(), accession_number)) as cursor:
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"], "form_type": r["form_type"],
                "analysis_type": r["analysis_type"],
                "analysis_json": r["analysis_json"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
