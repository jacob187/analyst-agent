FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ../pyproject.toml .
RUN pip install --no-cache-dir \
    edgartools>=3.14.2 \
    python-dotenv>=1.1.0 \
    langchain>=0.3.22 \
    langchain-community>=0.3.20 \
    langchain-google-genai>=2.1.4 \
    langgraph>=0.6.4 \
    yfinance>=0.2.55

# Copy application code
COPY . /app/agents
COPY ../database /app/database

ENV PYTHONPATH=/app

CMD ["python", "-m", "agents.main_agent"]
