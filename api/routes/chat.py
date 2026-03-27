"""WebSocket chat endpoint — real-time agent interaction."""

import asyncio

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

        # Resolve session
        explicit_session_id = auth_message.get("session_id")
        if explicit_session_id:
            session_id = explicit_session_id
        else:
            session_id = await get_or_create_session(ticker)

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

        # Initialise agent
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

        # Chat loop
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
