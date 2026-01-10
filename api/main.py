from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

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
    await websocket.send_json({"message": f"Connected to {ticker}"})

    while True:
        try:
            data = await websocket.receive_text()
            await websocket.send_json({"message": f"Echo: {data}"})
        except:
            break
