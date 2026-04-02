"""FastAPI application — thin shell that wires routers together.

Business logic lives in:
- api/routes/health.py    — health check
- api/routes/sessions.py  — session CRUD
- api/routes/chart.py     — stock chart data (OHLCV + indicators)
- api/routes/chat.py      — WebSocket chat with LLM agent
- api/memory.py           — conversation compression + context reconstruction
- api/db.py               — SQLite database layer
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.db import init_db, close_db
from api.routes.health import router as health_router
from api.routes.sessions import router as sessions_router
from api.routes.chart import router as chart_router
from api.routes.chat import router as chat_router
from api.routes.watchlist import router as watchlist_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(title="Analyst Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(chart_router)
app.include_router(chat_router)
app.include_router(watchlist_router)
