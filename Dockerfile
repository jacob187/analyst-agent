# Railway-compatible Dockerfile for API service
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies only (don't build the project itself)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY api ./api
COPY agents ./agents

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8000
EXPOSE 8000

CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
