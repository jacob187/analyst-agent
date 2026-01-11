from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(title="Analyst Agent API")

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

        await websocket.send_json({
            "type": "auth_success",
            "message": f"Connected to {ticker}. Ready to analyze."
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

            agent = create_sec_qa_agent(ticker, llm)

            await websocket.send_json({
                "type": "system",
                "message": f"Agent initialized for {ticker}"
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
