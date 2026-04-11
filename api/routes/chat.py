"""WebSocket chat endpoint — real-time agent interaction."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, AIMessage

from api.db import (
    get_or_create_session, get_session,
    save_message, get_session_messages,
)
from api.memory import (
    estimate_tokens, compress_history, build_context_from_session,
)
from api.dependencies import resolve_ws_keys
from api.rate_limit import check_rate_limit
from agents.model_registry import get_model, get_default_model, get_token_threshold
from agents.llm_factory import create_llm_pair

logger = logging.getLogger(__name__)

router = APIRouter()


async def _safe_send(websocket: WebSocket, data: dict) -> bool:
    """Send JSON, returning False if the connection is already closed."""
    try:
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@router.websocket("/ws/chat/{ticker}")
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

        keys = resolve_ws_keys(auth_message)

        if not keys.sec_header:
            await websocket.send_json({
                "type": "error",
                "message": "sec_header is required"
            })
            await websocket.close()
            return

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
                "message": f"{provider_display} API key required for {model.display_name}. Add it in Settings."
            })
            await websocket.close()
            return

        # Resolve session
        explicit_session_id = auth_message.get("session_id")
        if explicit_session_id:
            session_id = explicit_session_id
        else:
            session_id = await get_or_create_session(ticker, model=model_id)

        session = await get_session(session_id)

        # Validate that the session belongs to the ticker in the URL.
        # Without this check, a client could supply any session_id and
        # read another ticker's full conversation history (IDOR).
        if session is None or session["ticker"].upper() != ticker.upper():
            await websocket.send_json({"type": "error", "message": "Invalid session."})
            await websocket.close()
            return

        all_messages = await get_session_messages(session_id)
        conversation_history = build_context_from_session(session, all_messages)

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
                sec_header=keys.sec_header,
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
        client_ip = websocket.client.host if websocket.client else "unknown"
        while True:
            try:
                message = await websocket.receive_json()

                if message.get("type") == "query":
                    if not check_rate_limit(client_ip):
                        await websocket.send_json({
                            "type": "error",
                            "message": "Rate limited — please wait before sending more messages"
                        })
                        continue

                    user_query = message.get("message", "")

                    asyncio.create_task(save_message(session_id, "user", user_query))
                    conversation_history.append(HumanMessage(content=user_query))

                    full_response = ""
                    try:
                        async for event in agent.stream(
                            {"messages": conversation_history}
                        ):
                            if not await _safe_send(websocket, event):
                                return  # client disconnected mid-stream
                            if event["type"] == "response":
                                full_response = event["message"]
                    except WebSocketDisconnect:
                        return
                    except Exception as e:
                        logger.error("Error processing query for %s: %s", ticker, e, exc_info=True)
                        await _safe_send(websocket, {
                            "type": "error",
                            "message": "Error processing your query. Please try again."
                        })

                    if full_response:
                        conversation_history.append(AIMessage(content=full_response))
                        asyncio.create_task(
                            save_message(session_id, "assistant", full_response)
                        )

                        if estimate_tokens(conversation_history) > token_threshold:
                            conversation_history = await compress_history(
                                session_id, conversation_history, llm
                            )

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
