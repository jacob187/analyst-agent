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
    return _db

async def init_db():
    """Initialize database and create tables if they don't exist."""
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration: add summary column to existing databases that predate this column
    try:
        await db.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
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
    # NOTE: settings table removed — API keys now live in browser localStorage
    # and are sent per-request (WebSocket auth message or request headers).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    # Migration: backfill companies from existing watchlist rows
    await db.execute("""
        INSERT OR IGNORE INTO companies (ticker, added_at)
        SELECT ticker, added_at FROM watchlist
    """)

    await db.commit()

async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None

async def create_session(ticker: str) -> str:
    """Create a new chat session. Returns session ID.

    Also ensures a companies row exists for the ticker.
    """
    session_id = str(uuid.uuid4())
    db = await get_db()
    ticker = ticker.upper()
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.execute(
        "INSERT INTO sessions (id, ticker) VALUES (?, ?)",
        (session_id, ticker)
    )
    await db.commit()
    return session_id

async def get_or_create_session(ticker: str) -> str:
    """
    Return the existing session ID for this ticker, or create one.

    One session per ticker is enforced at the application level. This means
    entering AAPL always resumes the AAPL conversation — delete the session
    from History to start fresh.
    """
    db = await get_db()
    ticker = ticker.upper()
    async with db.execute(
        "SELECT id FROM sessions WHERE ticker = ?", (ticker,)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return row["id"]
    # No existing session — create one (and ensure company exists)
    session_id = str(uuid.uuid4())
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.execute(
        "INSERT INTO sessions (id, ticker) VALUES (?, ?)", (session_id, ticker)
    )
    await db.commit()
    return session_id

async def get_session_by_ticker(ticker: str) -> dict | None:
    """Return session metadata for a ticker, or None if no session exists."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, created_at FROM sessions WHERE ticker = ?",
        (ticker.upper(),)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return {"id": row["id"], "ticker": row["ticker"], "created_at": row["created_at"]}
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

async def get_sessions(limit: int = 50) -> list[dict]:
    """Get recent sessions."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ) as cursor:
        rows = await cursor.fetchall()
        return [{"id": row["id"], "ticker": row["ticker"], "created_at": row["created_at"]} for row in rows]

async def get_session(session_id: str) -> dict | None:
    """Get a specific session by ID, including any stored summary."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, summary, created_at FROM sessions WHERE id = ?",
        (session_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "ticker": row["ticker"],
                "summary": row["summary"],
                "created_at": row["created_at"],
            }
        return None


async def get_watchlist() -> list[dict]:
    """Return all watchlist tickers ordered by added_at."""
    db = await get_db()
    async with db.execute("SELECT ticker, added_at FROM watchlist ORDER BY added_at") as cursor:
        rows = await cursor.fetchall()
        return [{"ticker": row["ticker"], "added_at": row["added_at"]} for row in rows]


async def add_to_watchlist(ticker: str) -> bool:
    """Add ticker. Returns False if already present or limit (10) reached.

    Also ensures a companies row exists for the ticker (upsert).
    """
    db = await get_db()
    ticker = ticker.upper()
    # Check limit
    async with db.execute("SELECT COUNT(*) as cnt FROM watchlist") as cursor:
        row = await cursor.fetchone()
        if row["cnt"] >= 10:
            return False
    # Insert (ignore duplicate)
    cursor = await db.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker,))
    # Ensure company record exists
    await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (ticker,))
    await db.commit()
    return cursor.rowcount > 0


async def remove_from_watchlist(ticker: str) -> bool:
    """Remove ticker. Returns True if deleted."""
    db = await get_db()
    cursor = await db.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
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


async def get_company_activity(ticker: str) -> dict:
    """Return combined activity timeline for a company: sessions + briefings."""
    db = await get_db()
    ticker = ticker.upper()

    async with db.execute(
        "SELECT id, created_at FROM sessions WHERE ticker = ? ORDER BY created_at DESC",
        (ticker,)
    ) as cursor:
        sessions = [{"id": r["id"], "created_at": r["created_at"]} for r in await cursor.fetchall()]

    async with db.execute("""
        SELECT b.id, b.market_regime, b.created_at, bt.outlook
        FROM briefing_tickers bt
        JOIN briefings b ON bt.briefing_id = b.id
        WHERE bt.ticker = ?
        ORDER BY b.created_at DESC
    """, (ticker,)) as cursor:
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
                        tickers: list[dict]) -> str:
    """Persist a briefing and its per-ticker data. Returns briefing ID.

    Args:
        raw_json: Full DailyBriefingAnalysis.model_dump_json()
        market_regime: Regime summary string
        market_positioning: Positioning summary string
        alerts_json: JSON-serialized list of alert strings
        thinking: LLM chain-of-thought (may be empty)
        tickers: List of dicts with keys: ticker, price, change_pct,
                 technical_signal, news_summary, news_url, outlook
    """
    briefing_id = str(uuid.uuid4())
    db = await get_db()
    await db.execute("""
        INSERT INTO briefings (id, market_regime, market_positioning, alerts, thinking, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (briefing_id, market_regime, market_positioning, alerts_json, thinking, raw_json))

    for t in tickers:
        ticker_id = str(uuid.uuid4())
        # Defensive: ensure company exists before FK insert
        await db.execute("INSERT OR IGNORE INTO companies (ticker) VALUES (?)", (t["ticker"],))
        await db.execute("""
            INSERT INTO briefing_tickers
                (id, briefing_id, ticker, price, change_pct, technical_signal,
                 news_summary, news_url, outlook)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker_id, briefing_id, t["ticker"], t["price"], t["change_pct"],
            t["technical_signal"], t["news_summary"], t.get("news_url"),
            t["outlook"],
        ))

    await db.commit()
    return briefing_id


async def get_recent_briefings(limit: int = 10) -> list[dict]:
    """Return most recent briefings with their tickers."""
    db = await get_db()
    async with db.execute(
        "SELECT id, market_regime, market_positioning, alerts, thinking, created_at "
        "FROM briefings ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ) as cursor:
        rows = await cursor.fetchall()

    results = []
    for row in rows:
        briefing = {
            "id": row["id"],
            "market_regime": row["market_regime"],
            "market_positioning": row["market_positioning"],
            "alerts": row["alerts"],
            "thinking": row["thinking"],
            "created_at": row["created_at"],
            "tickers": [],
        }
        async with db.execute(
            "SELECT ticker, price, change_pct, technical_signal, news_summary, "
            "news_url, outlook FROM briefing_tickers WHERE briefing_id = ?",
            (row["id"],)
        ) as tc:
            ticker_rows = await tc.fetchall()
            briefing["tickers"] = [
                {
                    "ticker": tr["ticker"], "price": tr["price"],
                    "change_pct": tr["change_pct"], "technical_signal": tr["technical_signal"],
                    "news_summary": tr["news_summary"], "news_url": tr["news_url"],
                    "outlook": tr["outlook"],
                }
                for tr in ticker_rows
            ]
        results.append(briefing)
    return results


async def get_briefing_history(ticker: str, days: int = 30) -> list[dict]:
    """Return briefing entries for a specific ticker within the last N days."""
    db = await get_db()
    ticker = ticker.upper()
    async with db.execute("""
        SELECT b.id, b.market_regime, b.created_at,
               bt.price, bt.change_pct, bt.technical_signal,
               bt.news_summary, bt.outlook
        FROM briefing_tickers bt
        JOIN briefings b ON bt.briefing_id = b.id
        WHERE bt.ticker = ?
          AND b.created_at >= datetime('now', ? || ' days')
        ORDER BY b.created_at DESC
    """, (ticker, f"-{days}")) as cursor:
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
