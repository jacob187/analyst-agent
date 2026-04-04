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
    TOKEN_THRESHOLD,
)
from api.dependencies import resolve_ws_keys
from api.rate_limit import check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()


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

        if not keys.google_api_key or not keys.sec_header:
            await websocket.send_json({
                "type": "error",
                "message": "Both google_api_key and sec_header are required"
            })
            await websocket.close()
            return

        # Resolve session
        explicit_session_id = auth_message.get("session_id")
        if explicit_session_id:
            session_id = explicit_session_id
        else:
            session_id = await get_or_create_session(ticker)

        session = await get_session(session_id)
        all_messages = await get_session_messages(session_id)
        conversation_history = build_context_from_session(session, all_messages)

        research_status = "Web research enabled" if keys.tavily_api_key else "Web research disabled (no Tavily API key)"
        await websocket.send_json({
            "type": "auth_success",
            "message": f"Connected to {ticker}. {research_status}. Ready to analyze.",
            "session_id": session_id,
            "resumed": len(all_messages) > 0,
        })

        # Initialise agent
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from agents.graph.analyst_graph import create_sec_qa_agent

            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=keys.google_api_key,
                temperature=0,
                thinking_level="low",
            )
            synthesizer_llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=keys.google_api_key,
                temperature=0,
                thinking_level="medium",
                include_thoughts=True,
            )

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
                            await websocket.send_json(event)
                            if event["type"] == "response":
                                full_response = event["message"]
                    except Exception as e:
                        logger.error("Error processing query for %s: %s", ticker, e, exc_info=True)
                        await websocket.send_json({
                            "type": "error",
                            "message": "Error processing your query. Please try again."
                        })

                    if full_response:
                        conversation_history.append(AIMessage(content=full_response))
                        asyncio.create_task(
                            save_message(session_id, "assistant", full_response)
                        )

                        if estimate_tokens(conversation_history) > TOKEN_THRESHOLD:
                            conversation_history = await compress_history(
                                session_id, conversation_history, llm
                            )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Unexpected error in chat loop for %s: %s", ticker, e, exc_info=True)
                await websocket.send_json({
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
