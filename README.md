# Stock Analyst Agent

A full-stack AI-powered financial analysis application that combines SEC filings analysis, technical indicators, and real-time web research through an interactive chat interface.

## Tech Stack

### Frontend
- **Svelte 5** - Reactive UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server

### Backend
- **FastAPI** - Python async web framework
- **WebSocket** - Real-time bidirectional communication
- **LangGraph** - Graph-based AI agent orchestration
- **LangChain** - LLM framework with Google Gemini

### Data Sources
- **SEC EDGAR** - 10-K and 10-Q filings via edgartools
- **Yahoo Finance** - Market data and technical indicators
- **Tavily** - Web research and news (optional)

## Features

- **Interactive Chat Interface**: Real-time WebSocket communication with AI agent
- **SEC Filings Analysis**: Extracts and analyzes MD&A, Risk Factors, and Balance Sheets
- **Technical Analysis**: RSI, MACD, Bollinger Bands, moving averages, and more
- **Web Research**: Company news, competitor analysis, and industry trends (with Tavily)
- **ReAct Agent Pattern**: Reason → Act → Observe → Synthesize workflow
- **Smart Caching**: Minimizes redundant API calls for efficient data retrieval

## Installation

### Prerequisites
- Python 3.12+
- Node.js 20+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Backend Setup

```bash
# Install Python dependencies
uv sync
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the application

```bash
./start.sh
```

This runs both the backend (port 8000) and frontend (port 5173).

### 2. Enter API keys in the UI

Navigate to Settings in the app and enter your API keys:
- **Google API Key** - For Gemini LLM
- **SEC Header** - Your email (SEC requires identification)
- **Tavily API Key** - Optional, for web research

### 3. Open in browser

- Frontend: http://localhost:5173
- API: http://localhost:8000

### Frontend Commands

```bash
npm run dev      # Development server with hot reload
npm run build    # Production build
npm run preview  # Preview production build
npm run check    # Type checking and linting
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ws/chat/{ticker}` | WebSocket | Chat interface for stock analysis |

### WebSocket Message Types

- `auth` - Authentication with API keys
- `query` - User question about the stock
- `response` - AI agent response
- `status` - Agent status updates
- `error` - Error messages

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│    Frontend     │◄──────────────────►│    FastAPI      │
│    (Svelte)     │                    │    Backend      │
└─────────────────┘                    └────────┬────────┘
                                                │
                                       ┌────────▼────────┐
                                       │  LangGraph      │
                                       │  ReAct Agent    │
                                       └────────┬────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
           ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
           │   SEC Tools     │        │  Stock Tools    │        │ Research Tools  │
           │  (edgartools)   │        │  (yfinance)     │        │   (Tavily)      │
           └─────────────────┘        └─────────────────┘        └─────────────────┘
```

## Available Tools

### SEC Filing Tools
- `get_risk_factors_summary` - Analyzed risks with sentiment scores
- `get_mda_summary` - MD&A analysis with outlook and key points
- `get_balance_sheet_summary` - Financial health with red flags
- `get_all_summaries` - Comprehensive overview

### Stock Market Tools
- `get_stock_price_history` - Last 10 trading days OHLC data
- `get_technical_analysis` - RSI, MACD, Bollinger Bands, moving averages
- `get_stock_info` - Current price, P/E, market cap, 52-week range

### Research Tools (Requires Tavily API Key)
- `web_search` - General company information search
- `deep_research` - Multi-source comprehensive research
- `get_company_news` - Latest news and developments
- `analyze_competitors` - Market positioning analysis
- `get_industry_trends` - Industry outlook forecasts

## Project Structure

```
analyst-agent/
├── api/                        # FastAPI backend
│   ├── main.py                # WebSocket and HTTP endpoints
│   └── db.py                  # SQLite database for chat history
├── agents/                     # AI agent implementation
│   ├── graph/
│   │   └── sec_graph.py       # LangGraph agent orchestration
│   ├── sec_workflow/          # SEC filings processing
│   │   ├── get_SEC_data.py
│   │   └── sec_llm_models.py
│   ├── technical_workflow/    # Stock data processing
│   │   ├── get_stock_data.py
│   │   └── process_technical_indicators.py
│   └── tools/                 # Tool definitions
│       ├── sec_tools.py       # SEC filing tools
│       └── research_tools.py  # Tavily research tools
├── frontend/                   # Svelte frontend
│   ├── src/
│   │   ├── App.svelte
│   │   └── lib/components/
│   │       ├── ChatWindow.svelte
│   │       ├── ChatHistory.svelte
│   │       ├── ChatViewer.svelte
│   │       ├── ApiKeyInput.svelte
│   │       └── TickerInput.svelte
│   └── package.json
├── start.sh                    # Start script for local development
└── pyproject.toml              # Python dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Edgar Tools](https://github.com/dgunning/edgartools) - SEC filings retrieval
- [Yahoo Finance](https://pypi.org/project/yfinance/) - Market data
- [LangChain](https://python.langchain.com/) - LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent framework
- [Tavily](https://tavily.com/) - Web research API
- [Svelte](https://svelte.dev/) - Frontend framework
