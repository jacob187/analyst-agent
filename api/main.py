from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.graph.sec_graph import create_sec_agent

load_dotenv()

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

    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY"),
        )

        # Set up memory for conversation persistence
        memory = MemorySaver()

        # Create agent with tools for the ticker
        agent, llm_id = create_sec_agent(llm, ticker.upper(), checkpointer=memory)

        # Config for memory
        config = {"configurable": {"thread_id": f"chat_{ticker.lower()}"}}

        await websocket.send_json({"message": f"Connected to {ticker}"})

        while True:
            try:
                # Receive user message
                data = await websocket.receive_text()

                # Process with LangGraph agent
                messages = [HumanMessage(content=data)]
                result = agent.invoke({"messages": messages}, config=config)

                # Extract and send the response
                if result and "messages" in result:
                    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                    if ai_messages:
                        content = ai_messages[-1].content
                        # Handle different content types (string, list, dict)
                        if isinstance(content, str):
                            response = content
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            response = "\n".join(
                                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                                for block in content
                            )
                        else:
                            response = str(content)
                        await websocket.send_json({"message": response})
                    else:
                        await websocket.send_json({"message": "I processed your request but didn't generate a response."})
                else:
                    await websocket.send_json({"message": "No response generated"})

            except Exception as e:
                await websocket.send_json({"message": f"Error processing message: {str(e)}"})

    except ValueError as e:
        await websocket.send_json({"message": f"Error: {str(e)}"})
        await websocket.close()
    except Exception as e:
        await websocket.send_json({"message": f"Failed to initialize: {str(e)}"})
        await websocket.close()
