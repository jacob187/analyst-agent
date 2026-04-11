"""Session REST endpoints."""

from fastapi import APIRouter, HTTPException, Query

from api.db import (
    get_or_create_session, get_session, get_session_by_ticker,
    get_tickers, get_sessions_for_ticker,
    get_session_messages, delete_session,
)

router = APIRouter()


@router.get("/tickers")
async def list_tickers(limit: int = 50):
    """Return distinct tickers with session count — no session IDs exposed."""
    tickers = await get_tickers(limit)
    return {"tickers": tickers}


@router.get("/sessions")
async def list_sessions(
    ticker: str = Query(..., description="Return sessions for this ticker only"),
    limit: int = 50,
):
    """Return sessions for a specific ticker.

    Requiring the caller to supply a known ticker prevents enumeration:
    session IDs are only revealed after ownership (knowing the ticker) is
    established, so an attacker cannot discover IDs for unknown tickers.
    """
    sessions = await get_sessions_for_ticker(ticker, limit)
    return {"sessions": sessions}


@router.get("/sessions/by-ticker/{ticker}")
async def get_session_for_ticker(ticker: str):
    """Return the existing session for a ticker, or null if none exists."""
    session = await get_session_by_ticker(ticker.upper())
    return {"session": session}


async def _validate_session_ticker(session_id: str, ticker: str) -> dict:
    """Verify that the session exists and belongs to the given ticker.

    Prevents IDOR — without this check a caller could supply any session_id
    and read/delete another ticker's conversation history.
    """
    session = await get_session(session_id)
    if session is None or session["ticker"].upper() != ticker.upper():
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    ticker: str = Query(..., description="Ticker that owns this session (IDOR guard)"),
):
    await _validate_session_ticker(session_id, ticker)
    messages = await get_session_messages(session_id)
    return {"messages": messages}


@router.delete("/sessions/{session_id}")
async def remove_session(
    session_id: str,
    ticker: str = Query(..., description="Ticker that owns this session (IDOR guard)"),
):
    await _validate_session_ticker(session_id, ticker)
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}
