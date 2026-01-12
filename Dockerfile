# Railway-compatible Dockerfile for API service
# Build from repo root to access all directories

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.27.0 \
    python-dotenv>=1.1.0 \
    pydantic>=2.6.0 \
    pydantic-settings>=2.1.0 \
    edgartools>=3.14.2 \
    langchain>=0.3.22 \
    langchain-google-genai>=2.1.4 \
    langgraph>=0.6.4 \
    yfinance>=0.2.55

# Copy application code from repo root
COPY api /app/api
COPY agents /app/agents
COPY database /app/database

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
