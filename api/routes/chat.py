"""WebSocket chat endpoint — real-time agent interaction."""

import asyncio
import json
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, AIMessage

from api.db import (
    get_or_create_session, get_session,
    save_message, get_session_messages,
)
from api.memory import (
    estimate_tokens_incremental, compress_history, build_context_from_session,
)
from api.dependencies import resolve_ws_keys, verify_ws_identity
from api.llm_concurrency import (
    LLMBudgetExceeded,
    LLMConcurrencyExceeded,
    check_and_charge_budget,
)
from api.rate_limit import check_rate_limit
from api.validators import TICKER_RE
from agents.graph.analyst_graph import LLMTimeoutError
from agents.model_registry import get_model, get_default_model, get_token_threshold
from agents.llm_factory import create_llm_pair

logger = logging.getLogger(__name__)

router = APIRouter()

# WS hardening knobs — env-overridable so prod can tune and tests can run fast.
# These are read at call time via module attribute lookup so monkeypatch works.
WS_AUTH_TIMEOUT_SECONDS = float(os.getenv("WS_AUTH_TIMEOUT_SECONDS", "30"))
WS_AUTH_MAX_BYTES = int(os.getenv("WS_AUTH_MAX_BYTES", "16384"))
WS_IDLE_TIMEOUT_SECONDS = float(os.getenv("WS_IDLE_TIMEOUT_SECONDS", "300"))
WS_SEND_TIMEOUT_SECONDS = float(os.getenv("WS_SEND_TIMEOUT_SECONDS", "10"))

MAX_QUERY_LENGTH = 4000


async def _safe_send(websocket: WebSocket, data: dict) -> bool:
    """Send JSON with a send timeout; return False if the peer is gone or slow.

    A hanging `send_json` would otherwise pin the agent task on a slow consumer
    (full TCP buffer, dead connection that hasn't been reaped yet).
    """
    try:
        await asyncio.wait_for(
            websocket.send_json(data),
            timeout=WS_SEND_TIMEOUT_SECONDS,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning("WS send timed out after %ss — closing slow consumer", WS_SEND_TIMEOUT_SECONDS)
        try:
            await websocket.close(code=1008)
        except Exception:
            pass
        return False
    except (WebSocketDisconnect, RuntimeError):
        return False


@router.websocket("/ws/chat/{ticker}")
async def chat(websocket: WebSocket, ticker: str):
    await websocket.accept()

    if not TICKER_RE.match(ticker.upper()):
        await websocket.send_json({"type": "error", "message": "Invalid ticker symbol."})
        await websocket.close()
        return

    try:
        # Size + time-bounded auth frame read. Plain `receive_json()` has no
        # ceiling — a client can stall indefinitely or push megabytes of JSON.
        try:
            raw_auth = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=WS_AUTH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1008)
            return

        if len(raw_auth) > WS_AUTH_MAX_BYTES:
            await websocket.close(code=1008)
            return

        try:
            auth_message = json.loads(raw_auth)
        except (json.JSONDecodeError, ValueError):
            await websocket.send_json({"type": "error", "message": "Invalid auth payload"})
            await websocket.close(code=1008)
            return

        if not isinstance(auth_message, dict):
            await websocket.send_json({"type": "error", "message": "Auth payload must be an object"})
            await websocket.close(code=1008)
            return

        if auth_message.get("type") != "auth":
            await websocket.send_json({
                "type": "error",
                "message": "First message must be authentication with API keys"
            })
            await websocket.close()
            return

        keys = resolve_ws_keys(auth_message)

        ok, reason = verify_ws_identity(keys.user_id, auth_message)
        if not ok:
            await websocket.send_json({"type": "error", "message": reason or "Unauthorized"})
            await websocket.close(code=1008)
            return

        # Anonymous (no user_id) is allowed: signed-out visitors may chat with
        # their own BYOK key, ephemerally. Persistence is sign-in-only (below).
        user_id = keys.user_id

        # Resolve model from auth message → registry default
        model_id = keys.model_id or get_default_model().id
        model = get_model(model_id)
        if model is None:
            default = get_default_model()
            model = default
            model_id = default.id

        # Resolve the API key for the selected provider
        api_key = keys.get_provider_key(model.provider)
        if not api_key:
            provider_display = model.provider.replace("_", " ").title()
            await websocket.send_json({
                "type": "error",
                "message": f"{provider_display} API key required for {model.display_name}. Sign in or add your key in Settings."
            })
            await websocket.close()
            return

        # If the operator's env key resolves the LLM (not BYOK), each user
        # message draws against the per-user daily budget. The model is fixed
        # for the session, so resolve once and reuse on every message.
        operator_paid = keys.is_operator_paid(model.provider)

        # Persistence is sign-in-only. Anonymous users (BYOK) get an ephemeral,
        # in-memory conversation — no session row, no saved messages.
        if user_id:
            explicit_session_id = auth_message.get("session_id")
            if explicit_session_id:
                session_id = explicit_session_id
            else:
                session_id = await get_or_create_session(ticker, user_id, model=model_id)

            session = await get_session(session_id)

            # Validate session belongs to this ticker AND this user (IDOR guard).
            if (
                session is None
                or session["ticker"].upper() != ticker.upper()
                or session.get("user_id") != user_id
            ):
                await websocket.send_json({"type": "error", "message": "Invalid session."})
                await websocket.close()
                return

            all_messages = await get_session_messages(session_id)
            conversation_history = build_context_from_session(session, all_messages)
        else:
            session_id = None
            all_messages = []
            conversation_history = []

        research_status = "Web research enabled" if keys.tavily_api_key else "Web research disabled (no Tavily API key)"
        await websocket.send_json({
            "type": "auth_success",
            "message": f"Connected to {ticker} using {model.display_name}. {research_status}. Ready to analyze.",
            "session_id": session_id,
            "resumed": len(all_messages) > 0,
            "model_id": model_id,
        })

        # Initialise agent
        try:
            from agents.graph.analyst_graph import create_sec_qa_agent

            llm, synthesizer_llm = create_llm_pair(model_id, api_key)

            agent = create_sec_qa_agent(
                ticker, llm,
                tavily_api_key=keys.tavily_api_key,
                synthesizer_llm=synthesizer_llm,
                user_id=user_id,
            )

            tools_count = "16 tools" if keys.tavily_api_key else "11 tools"
            await websocket.send_json({
                "type": "system",
                "message": f"Planning agent initialized for {ticker} with {tools_count}. Complex queries will be auto-decomposed."
            })

        except Exception as e:
            logger.error("Failed to initialize agent for %s: %s", ticker, e, exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": "Failed to initialize agent. Check your API keys and try again."
            })
            await websocket.close()
            return

        # Chat loop
        token_threshold = get_token_threshold(model_id)
        # Incremental token tally — avoids re-walking the full history each turn.
        # estimate_tokens_incremental walks only new messages instead of all,
        # turning the cumulative session cost O(N²) → O(N). `chars_so_far` is
        # the running character total threaded between turns; tokens are
        # derived inside the helper.
        chars_so_far, messages_counted = 0, 0
        client_ip = websocket.client.host if websocket.client else "unknown"
        while True:
            try:
                # Reap idle sessions — clients that disappear without sending FIN
                # would otherwise keep an agent + LLM client + history in memory
                # for the lifetime of the process.
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=WS_IDLE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.info("Idle WS session closed after %ss (%s)", WS_IDLE_TIMEOUT_SECONDS, ticker)
                    try:
                        await websocket.close(code=1001)  # going away
                    except Exception:
                        pass
                    break

                if message.get("type") == "query":
                    if not check_rate_limit(client_ip):
                        await websocket.send_json({
                            "type": "error",
                            "message": "Rate limited — please wait before sending more messages"
                        })
                        continue

                    if operator_paid:
                        try:
                            await check_and_charge_budget(user_id)
                        except LLMBudgetExceeded:
                            await _safe_send(websocket, {
                                "type": "error",
                                "message": "Daily LLM budget reached. Try again tomorrow or add your own API key in Settings.",
                            })
                            continue

                    user_query = message.get("message", "")

                    if len(user_query) > MAX_QUERY_LENGTH:
                        await _safe_send(websocket, {
                            "type": "error",
                            "message": f"Message too long (max {MAX_QUERY_LENGTH} characters).",
                        })
                        continue

                    if session_id:
                        asyncio.create_task(save_message(session_id, "user", user_query))
                    conversation_history.append(HumanMessage(content=user_query))

                    full_response = ""
                    try:
                        async for event in agent.stream(
                            {"messages": conversation_history},
                            config={
                                "metadata": {
                                    "session_id": session_id,
                                    "ticker": ticker,
                                    "user_id": user_id,
                                },
                                "run_name": f"chat:{ticker}",
                            },
                        ):
                            if not await _safe_send(websocket, event):
                                return  # client disconnected mid-stream
                            if event["type"] == "response":
                                full_response = event["message"]
                    except WebSocketDisconnect:
                        return
                    except LLMTimeoutError as e:
                        logger.warning("LLM timeout for %s: %s", ticker, e)
                        await _safe_send(websocket, {
                            "type": "error",
                            "message": "The model took too long to respond. Please try a simpler query or try again.",
                        })
                    except LLMConcurrencyExceeded:
                        logger.warning("LLM dispatch pool saturated for %s", ticker)
                        await _safe_send(websocket, {
                            "type": "error",
                            "message": "Server busy — please retry shortly.",
                        })
                    except Exception as e:
                        logger.error("Error processing query for %s: %s", ticker, e, exc_info=True)
                        await _safe_send(websocket, {
                            "type": "error",
                            "message": "Error processing your query. Please try again."
                        })

                    if full_response:
                        conversation_history.append(AIMessage(content=full_response))
                        if session_id:
                            asyncio.create_task(
                                save_message(session_id, "assistant", full_response)
                            )

                        tokens_so_far, chars_so_far, messages_counted = (
                            estimate_tokens_incremental(
                                conversation_history, chars_so_far, messages_counted
                            )
                        )

                        if tokens_so_far > token_threshold:
                            conversation_history = await compress_history(
                                session_id, conversation_history, llm
                            )
                            # Compression replaces history with a shorter list.
                            # Reset the running tally so the next turn re-baselines.
                            chars_so_far, messages_counted = 0, 0

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Unexpected error in chat loop for %s: %s", ticker, e, exc_info=True)
                await _safe_send(websocket, {
                    "type": "error",
                    "message": "An unexpected error occurred."
                })
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Connection error for %s: %s", ticker, e, exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Connection error."
            })
        except Exception:
            pass
