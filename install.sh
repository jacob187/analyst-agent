#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
DIM='\033[2m'
NC='\033[0m'

echo -e "${BLUE}Installing Analyst Agent...${NC}"

# Check prerequisites
missing=""
if ! command -v uv &> /dev/null; then missing="uv (https://docs.astral.sh/uv)"; fi
if ! command -v node &> /dev/null; then missing="$missing${missing:+, }Node.js 20+ (https://nodejs.org)"; fi

if [ -n "$missing" ]; then
    echo -e "${RED}Missing: ${missing}${NC}"
    echo "Install the above and re-run this script."
    exit 1
fi

# Python dependencies
echo -e "${DIM}Installing Python dependencies...${NC}"
uv sync

# Frontend dependencies
echo -e "${DIM}Installing frontend dependencies...${NC}"
cd frontend && npm ci && cd ..

# Environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}Created .env from .env.example — edit it with your API keys.${NC}"
else
    echo -e "${DIM}.env already exists, skipping.${NC}"
fi

# Create data directory for SQLite
mkdir -p data

echo ""
echo -e "${GREEN}Done.${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (at minimum: GOOGLE_API_KEY)"
echo "  2. Run ./start.sh to launch the app"
echo "  3. Open http://localhost:5173"
