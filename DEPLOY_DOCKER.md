# Docker Deployment Guide

Deploy the Analyst Agent app using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Google Gemini API key ([get one here](https://makersuite.google.com/app/apikey))

## Quick Start

### 1. Clone and checkout deploy branch

```bash
git clone <your-repo-url>
cd analyst-agent
git checkout deploy
```

### 2. Create environment file

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
GOOGLE_API_KEY="your_google_gemini_api_key"
SEC_HEADER="your.email@example.com YourName"
API_PORT=8000
FRONTEND_PORT=3000
```

> **Note:** `SEC_HEADER` is required by the SEC EDGAR API. Use format: `email@example.com Your Name`

### 3. Build and run

```bash
docker compose up --build -d
```

### 4. Verify deployment

```bash
# Check all containers are running
docker compose ps

# Check API health
curl http://localhost:8000/health
```

### 5. Access the app

- **Frontend:** http://localhost:3000
- **API:** http://localhost:8000
- **Health check:** http://localhost:8000/health

## Commands Reference

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start all services |
| `docker compose down` | Stop all services |
| `docker compose logs -f` | View logs (follow mode) |
| `docker compose logs api` | View API logs only |
| `docker compose restart api` | Restart API service |
| `docker compose build --no-cache` | Rebuild without cache |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    API Server   │
│  (Nginx:3000)   │     │  (FastAPI:8000) │
└─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │     Agents      │
                        │   (Workers)     │
                        └─────────────────┘
```

**Services:**
- `analyst-frontend` - Svelte SPA served by Nginx
- `analyst-api` - FastAPI backend with WebSocket support
- `analyst-agents` - Background agent workers

## Troubleshooting

### Container won't start
```bash
docker compose logs api
```

### WebSocket connection failed
- Ensure port 8000 is accessible
- Check CORS settings in `api/main.py`

### SEC API errors
- Verify `SEC_HEADER` format: `email@example.com YourName`

### Rebuild after code changes
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```
