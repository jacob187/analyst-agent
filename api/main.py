from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
import json
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.db import init_db, close_db, create_session, save_message, get_sessions, get_session_messages

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database
    await init_db()
    yield
    # Shutdown: close database connection
    await close_db()

app = FastAPI(title="Analyst Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent chat sessions."""
    sessions = await get_sessions(limit)
    return {"sessions": sessions}

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """Get all messages for a session."""
    messages = await get_session_messages(session_id)
    return {"messages": messages}

@app.websocket("/ws/chat/{ticker}")
async def chat(websocket: WebSocket, ticker: str):
    await websocket.accept()

    # Wait for authentication message with API keys
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
        tavily_api_key = auth_message.get("tavily_api_key")  # Optional

        if not google_api_key or not sec_header:
            await websocket.send_json({
                "type": "error",
                "message": "Both google_api_key and sec_header are required"
            })
            await websocket.close()
            return

        # Set environment variables for this session
        os.environ["GOOGLE_API_KEY"] = google_api_key
        os.environ["SEC_HEADER"] = sec_header
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key

        # Create a new chat session
        session_id = await create_session(ticker)

        research_status = "Web research enabled" if tavily_api_key else "Web research disabled (no Tavily API key)"
        await websocket.send_json({
            "type": "auth_success",
            "message": f"Connected to {ticker}. {research_status}. Ready to analyze.",
            "session_id": session_id
        })

        # Import and initialize LangGraph agent
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from agents.graph.sec_graph import create_sec_qa_agent

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=google_api_key,
                temperature=0
            )

            agent = create_sec_qa_agent(ticker, llm, tavily_api_key=tavily_api_key)

            tools_count = "16 tools" if tavily_api_key else "11 tools"
            await websocket.send_json({
                "type": "system",
                "message": f"Agent initialized for {ticker} with {tools_count}"
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

                    # Save user message (fire-and-forget)
                    asyncio.create_task(save_message(session_id, "user", user_query))

                    # Stream agent response
                    await websocket.send_json({
                        "type": "status",
                        "message": "Processing query..."
                    })

                    try:
                        # Invoke agent
                        result = agent.invoke({"messages": [("user", user_query)]})

                        # Send response
                        if result and "messages" in result:
                            response = result["messages"][-1].content
                        else:
                            response = str(result)

                        # Save assistant response (fire-and-forget)
                        asyncio.create_task(save_message(session_id, "assistant", response))

                        await websocket.send_json({
                            "type": "response",
                            "message": response
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error processing query: {str(e)}"
                        })

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
