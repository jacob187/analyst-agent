"""Session REST endpoints."""

from fastapi import APIRouter, HTTPException

from api.db import (
    get_or_create_session, get_session_by_ticker,
    get_sessions, get_session_messages, delete_session,
)

router = APIRouter()


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    sessions = await get_sessions(limit)
    return {"sessions": sessions}


@router.get("/sessions/by-ticker/{ticker}")
async def get_session_for_ticker(ticker: str):
    """Return the existing session for a ticker, or null if none exists."""
    session = await get_session_by_ticker(ticker.upper())
    return {"session": session}


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    messages = await get_session_messages(session_id)
    return {"messages": messages}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}
