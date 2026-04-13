"""Session REST endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from api.validators import TICKER_RE
from api.dependencies import ApiKeys, get_api_keys
from api.db import (
    get_or_create_session, get_session, get_session_by_ticker,
    get_tickers, get_sessions_for_ticker,
    get_session_messages, delete_session,
)

router = APIRouter()


@router.get("/tickers")
async def list_tickers(limit: int = 50, keys: ApiKeys = Depends(get_api_keys)):
    """Return distinct tickers with session count for this user."""
    user_id = keys.require_user_id()
    tickers = await get_tickers(user_id, limit)
    return {"tickers": tickers}


@router.get("/sessions")
async def list_sessions(
    ticker: str = Query(..., description="Return sessions for this ticker only"),
    limit: int = 50,
    keys: ApiKeys = Depends(get_api_keys),
):
    """Return sessions for a specific ticker, scoped to the current user."""
    user_id = keys.require_user_id()
    sessions = await get_sessions_for_ticker(ticker, user_id, limit)
    return {"sessions": sessions}


@router.get("/sessions/by-ticker/{ticker}")
async def get_session_for_ticker(ticker: str, keys: ApiKeys = Depends(get_api_keys)):
    """Return the existing session for a ticker+user, or null if none exists."""
    if not TICKER_RE.match(ticker.upper()):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    user_id = keys.require_user_id()
    session = await get_session_by_ticker(ticker.upper(), user_id)
    return {"session": session}


async def _validate_session_ticker(session_id: str, ticker: str, user_id: str) -> dict:
    """Verify that the session exists, belongs to the given ticker, AND to the user.

    Prevents IDOR — without the user_id check a caller could supply any
    session_id and read/delete another user's conversation history.
    """
    session = await get_session(session_id)
    if (
        session is None
        or session["ticker"].upper() != ticker.upper()
        or session.get("user_id") != user_id
    ):
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    ticker: str = Query(..., description="Ticker that owns this session (IDOR guard)"),
    keys: ApiKeys = Depends(get_api_keys),
):
    user_id = keys.require_user_id()
    await _validate_session_ticker(session_id, ticker, user_id)
    messages = await get_session_messages(session_id)
    return {"messages": messages}


@router.delete("/sessions/{session_id}")
async def remove_session(
    session_id: str,
    ticker: str = Query(..., description="Ticker that owns this session (IDOR guard)"),
    keys: ApiKeys = Depends(get_api_keys),
):
    user_id = keys.require_user_id()
    await _validate_session_ticker(session_id, ticker, user_id)
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}
