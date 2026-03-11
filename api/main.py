from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
import json
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.db import (
    init_db, close_db,
    get_or_create_session, get_session, get_session_by_ticker,
    save_message, get_sessions, get_session_messages,
    get_settings, save_settings, delete_session, update_session_summary,
)

# How many tokens (rough estimate) in conversation_history before we compress.
# Gemini 3 Flash has a 1M token window, but tool call outputs can be large.
# 30K is conservative — leaves plenty of headroom for tool results and the
# agent's own reasoning tokens.
TOKEN_THRESHOLD = 30_000

# After compression, we keep this many recent messages verbatim so the LLM
# has full context for the current exchange. Everything older becomes a summary.
RECENT_MESSAGES_TO_KEEP = 6  # 3 user+assistant turns


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(title="Analyst Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list) -> int:
    """
    Rough token count using the ~4 chars/token heuristic.

    This is intentionally approximate — we're just deciding when to trigger
    compression, not billing by the token. The heuristic slightly over-estimates
    for code/numbers (fewer chars per token) and under-estimates for long
    natural language — which is fine since we have a generous threshold.
    """
    return sum(len(str(m.content)) for m in messages) // 4


async def compress_history(
    session_id: str,
    conversation_history: list,
    llm,
) -> list:
    """
    Summarize the older portion of conversation_history with the LLM, then
    return a shorter list: [summary_message] + last RECENT_MESSAGES_TO_KEEP.

    Full message history is always preserved in the messages table — this
    only affects what the LLM sees in its context window. The summary is
    persisted to the DB so it survives reconnects.
    """
    recent = conversation_history[-RECENT_MESSAGES_TO_KEEP:]
    to_summarize = conversation_history[:-RECENT_MESSAGES_TO_KEEP]

    # Build a readable transcript for the summarization call, collapsing
    # any prior summary messages so we don't nest summaries
    history_lines = []
    for msg in to_summarize:
        content = str(msg.content)
        if content.startswith("[CONVERSATION SUMMARY]"):
            history_lines.append(
                "PRIOR SUMMARY:\n" + content.replace("[CONVERSATION SUMMARY]", "").strip()
            )
        elif msg.type == "human":
            history_lines.append(f"USER: {content}")
        else:
            history_lines.append(f"ASSISTANT: {content}")

    prompt = (
        "You are compressing a financial analysis conversation for context management. "
        "Preserve all specific numbers, tickers, dates, analysis conclusions, open "
        "questions, and anything the user might want to reference later. Be concise "
        "but complete — this summary replaces the full history in future LLM calls.\n\n"
        + "\n\n".join(history_lines)
        + "\n\nSUMMARY:"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    summary_text = str(response.content)

    # Persist summary to DB asynchronously — full message history is untouched
    asyncio.create_task(update_session_summary(session_id, summary_text))

    summary_message = HumanMessage(content=f"[CONVERSATION SUMMARY]\n{summary_text}")
    return [summary_message] + recent


def build_context_from_session(
    session: dict,
    all_messages: list[dict],
) -> list:
    """
    Reconstruct conversation_history from DB state.

    If a summary exists and there are more messages than RECENT_MESSAGES_TO_KEEP,
    we use: [summary_message] + last N raw messages.
    Otherwise we load everything verbatim.

    This mirrors the in-session compression shape so the LLM never sees an
    inconsistent context structure on reconnect.
    """
    def to_lc(msg: dict):
        return (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )

    summary = session.get("summary") if session else None

    if summary and len(all_messages) > RECENT_MESSAGES_TO_KEEP:
        recent_raw = all_messages[-RECENT_MESSAGES_TO_KEEP:]
        return [
            HumanMessage(content=f"[CONVERSATION SUMMARY]\n{summary}")
        ] + [to_lc(m) for m in recent_raw]

    return [to_lc(m) for m in all_messages]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/sessions")
async def list_sessions(limit: int = 50):
    sessions = await get_sessions(limit)
    return {"sessions": sessions}


@app.get("/sessions/by-ticker/{ticker}")
async def get_session_for_ticker(ticker: str):
    """
    Return the existing session for a ticker symbol, or null if none exists.
    Used by the frontend to decide whether entering a ticker resumes an old
    conversation or starts a new one.
    """
    session = await get_session_by_ticker(ticker.upper())
    return {"session": session}


@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    messages = await get_session_messages(session_id)
    return {"messages": messages}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}


@app.get("/settings")
async def get_api_settings():
    settings = await get_settings()
    return {"settings": settings}


@app.post("/settings")
async def save_api_settings(data: dict):
    google_api_key = data.get("google_api_key")
    sec_header = data.get("sec_header")
    tavily_api_key = data.get("tavily_api_key")

    if not google_api_key or not sec_header:
        raise HTTPException(status_code=400, detail="google_api_key and sec_header are required")

    await save_settings(google_api_key, sec_header, tavily_api_key)
    return {"success": True}


# ---------------------------------------------------------------------------
# WebSocket chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat/{ticker}")
async def chat(websocket: WebSocket, ticker: str):
    await websocket.accept()

    try:
        auth_message = await websocket.receive_json()

        if auth_message.get("type") != "auth":
            await websocket.send_json({
                "type": "error",
                "message": "First message must be authentication with API keys"
            })
            await websocket.close()
            return

        google_api_key = auth_message.get("google_api_key")
        sec_header = auth_message.get("sec_header")
        tavily_api_key = auth_message.get("tavily_api_key")

        if not google_api_key or not sec_header:
            await websocket.send_json({
                "type": "error",
                "message": "Both google_api_key and sec_header are required"
            })
            await websocket.close()
            return

        # Resolve session -------------------------------------------------
        # If the client passes a session_id (e.g. "Continue" from History),
        # honour it. Otherwise get-or-create by ticker so there's always
        # exactly one session per ticker.
        explicit_session_id = auth_message.get("session_id")
        if explicit_session_id:
            session_id = explicit_session_id
        else:
            session_id = await get_or_create_session(ticker)

        # Build conversation history from DB state (summary + recent or all)
        session = await get_session(session_id)
        all_messages = await get_session_messages(session_id)
        conversation_history = build_context_from_session(session, all_messages)

        research_status = "Web research enabled" if tavily_api_key else "Web research disabled (no Tavily API key)"
        await websocket.send_json({
            "type": "auth_success",
            "message": f"Connected to {ticker}. {research_status}. Ready to analyze.",
            "session_id": session_id,
            "resumed": len(all_messages) > 0,
        })

        # Initialise agent ------------------------------------------------
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from agents.graph.sec_graph import create_sec_qa_agent

            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=google_api_key,
                temperature=0,
                thinking_level="low",
            )
            synthesizer_llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=google_api_key,
                temperature=0,
                thinking_level="medium",
                include_thoughts=True,
            )

            agent = create_sec_qa_agent(
                ticker, llm,
                tavily_api_key=tavily_api_key,
                synthesizer_llm=synthesizer_llm,
                sec_header=sec_header,
            )

            tools_count = "16 tools" if tavily_api_key else "11 tools"
            await websocket.send_json({
                "type": "system",
                "message": f"Planning agent initialized for {ticker} with {tools_count}. Complex queries will be auto-decomposed."
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize agent: {str(e)}"
            })
            await websocket.close()
            return

        # Chat loop -------------------------------------------------------
        while True:
            try:
                message = await websocket.receive_json()

                if message.get("type") == "query":
                    user_query = message.get("message", "")

                    asyncio.create_task(save_message(session_id, "user", user_query))
                    conversation_history.append(HumanMessage(content=user_query))

                    full_response = ""
                    try:
                        async for event in agent.stream(
                            {"messages": conversation_history}
                        ):
                            await websocket.send_json(event)
                            if event["type"] == "response":
                                full_response = event["message"]
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error processing query: {str(e)}"
                        })

                    if full_response:
                        conversation_history.append(AIMessage(content=full_response))
                        asyncio.create_task(
                            save_message(session_id, "assistant", full_response)
                        )

                        # Compress if context is getting large.
                        # compress_history() makes one LLM call and persists
                        # the summary to DB — full message history is untouched.
                        if estimate_tokens(conversation_history) > TOKEN_THRESHOLD:
                            conversation_history = await compress_history(
                                session_id, conversation_history, llm
                            )

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
