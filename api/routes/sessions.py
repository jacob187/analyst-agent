"""Session and settings REST endpoints."""

from fastapi import APIRouter, HTTPException

from api.db import (
    get_or_create_session, get_session_by_ticker,
    get_sessions, get_session_messages,
    get_settings, save_settings, delete_session,
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


@router.get("/settings")
async def get_api_settings():
    settings = await get_settings()
    return {"settings": settings}


@router.post("/settings")
async def save_api_settings(data: dict):
    google_api_key = data.get("google_api_key")
    sec_header = data.get("sec_header")
    tavily_api_key = data.get("tavily_api_key")

    if not google_api_key or not sec_header:
        raise HTTPException(status_code=400, detail="google_api_key and sec_header are required")

    await save_settings(google_api_key, sec_header, tavily_api_key)
    return {"success": True}
