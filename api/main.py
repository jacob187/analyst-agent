"""FastAPI application — thin shell that wires routers together.

Business logic lives in:
- api/routes/health.py    — health check
- api/routes/sessions.py  — session CRUD
- api/routes/chart.py     — stock chart data (OHLCV + indicators)
- api/routes/chat.py      — WebSocket chat with LLM agent
- api/routes/models.py    — available LLM models
- api/memory.py           — conversation compression + context reconstruction
- api/db.py               — SQLite database layer
"""

import logging
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env from project root so GOOGLE_API_KEY, TAVILY_API_KEY, etc.
# are available via os.getenv() as fallbacks for local development.
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.clerk_auth import is_auth_disabled, is_clerk_enabled
from api.db import init_db, close_db
from api.routes.health import router as health_router
from api.routes.sessions import router as sessions_router
from api.routes.chart import router as chart_router
from api.routes.chat import router as chat_router
from api.routes.watchlist import router as watchlist_router
from api.routes.models import router as models_router
from api.routes.company import router as company_router

logger = logging.getLogger(__name__)

_PROD_ENV_VALUES = ("production", "prod")


def check_production_auth_config() -> None:
    """Refuse to start in production when Clerk auth is disabled or unconfigured.

    Without this guard, `X-User-Id` is trusted as-is — full IDOR on
    sessions/watchlist/briefings. The `ANALYST_ALLOW_DISABLED_AUTH=1` escape
    hatch exists for emergency self-host operations.
    """
    env = (os.getenv("ENV") or "").strip().lower()
    logger.info("Startup env check: ENV=%r", env)
    if env not in _PROD_ENV_VALUES:
        return
    if (os.getenv("ANALYST_ALLOW_DISABLED_AUTH") or "").lower() in ("1", "true", "yes"):
        logger.warning("ANALYST_ALLOW_DISABLED_AUTH set — skipping production auth guard")
        return
    if is_auth_disabled():
        raise RuntimeError(
            "Refusing to start: ENV=production with DISABLE_AUTH=true. "
            "Unset DISABLE_AUTH, or set ANALYST_ALLOW_DISABLED_AUTH=1 to override."
        )
    if not is_clerk_enabled():
        raise RuntimeError(
            "Refusing to start: ENV=production but CLERK_SECRET_KEY is not set. "
            "Configure Clerk, or set ANALYST_ALLOW_DISABLED_AUTH=1 to override."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_production_auth_config()
    await init_db()
    yield
    await close_db()


app = FastAPI(title="Analyst Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=[
        "Content-Type",
        "X-Google-Api-Key",
        "X-Openai-Api-Key",
        "X-Anthropic-Api-Key",
        "X-Sec-Header",
        "X-Tavily-Api-Key",
        "X-Model-Id",
        "X-User-Id",
        "X-Clerk-Session-Token",
    ],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Convert ValueError (from require_user_id, require_provider_key) to 422."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(chart_router)
app.include_router(chat_router)
app.include_router(watchlist_router)
app.include_router(models_router)
app.include_router(company_router)
