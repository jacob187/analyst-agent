import aiosqlite
import uuid
from pathlib import Path

# Store db in project root/data directory
DB_PATH = Path(__file__).parent.parent / "data" / "chats.db"

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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
    await db.commit()

async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None

async def create_session(ticker: str) -> str:
    """Create a new chat session. Returns session ID."""
    session_id = str(uuid.uuid4())
    db = await get_db()
    await db.execute(
        "INSERT INTO sessions (id, ticker) VALUES (?, ?)",
        (session_id, ticker)
    )
    await db.commit()
    return session_id

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
    """Get a specific session by ID."""
    db = await get_db()
    async with db.execute(
        "SELECT id, ticker, created_at FROM sessions WHERE id = ?",
        (session_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            return {"id": row["id"], "ticker": row["ticker"], "created_at": row["created_at"]}
        return None
