FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (Docker layer caching — only re-installs when deps change)
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies from lockfile (no dev deps, no resolution)
RUN uv sync --frozen --no-dev

# Copy application source
COPY agents/ agents/
COPY api/ api/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
