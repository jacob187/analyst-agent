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
- **Intelligent Query Routing**: Automatically classifies queries and routes to optimal execution path
- **Multi-Step Planning**: Complex queries are decomposed into structured execution plans
- **SEC Filings Analysis**: Extracts and analyzes MD&A, Risk Factors, and Balance Sheets
- **Technical Analysis**: RSI, MACD, Bollinger Bands, moving averages, and more
- **Web Research**: Company news, competitor analysis, and industry trends (with Tavily)
- **Hybrid Agent Architecture**:
  - Simple queries → ReAct agent (dynamic tool selection)
  - Complex queries → Planner → Step-by-step execution → Synthesis
- **Smart Caching**: Minimizes redundant API calls for efficient data retrieval

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Run the Application

```bash
./start.sh
```

This installs dependencies (if needed) and starts both the backend (port 8000) and frontend (port 5173).

### Configure API Keys

Open http://localhost:5173 and enter your API keys in Settings:
- **Google API Key** - For Gemini LLM
- **SEC Header** - Your email (SEC requires identification)
- **Tavily API Key** - Optional, for web research

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

### Query Planning Workflow

The system uses a hybrid architecture that intelligently routes queries based on complexity:

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                  Query Router                         │
│  (Classifies complexity: simple vs. complex)          │
└───────────┬───────────────────────────────────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
  Simple      Complex
      │           │
      │           ▼
      │     ┌──────────────┐
      │     │   Planner    │ Creates multi-step execution plan
      │     └──────┬───────┘
      │            │
      │            ▼
      │     ┌──────────────┐
      │     │     Step     │ Executes each step with dependencies
      │     │   Executor   │ (loops until all steps complete)
      │     └──────┬───────┘
      │            │
      │            ▼
      │     ┌──────────────┐
      │     │ Synthesizer  │ Combines results into final response
      │     └──────┬───────┘
      │            │
      ▼            ▼
  ┌─────────────────┐
  │  ReAct Agent    │ Dynamic tool selection for simple queries
  └────────┬────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
 Response    Final Answer
```

### System Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│    Frontend     │◄──────────────────►│    FastAPI      │
│    (Svelte)     │                    │    Backend      │
└─────────────────┘                    └────────┬────────┘
                                                │
                                       ┌────────▼────────┐
                                       │  LangGraph      │
                                       │  Planning Agent │
                                       └────────┬────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
           ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
           │   SEC Tools     │        │  Stock Tools    │        │ Research Tools  │
           │  (edgartools)   │        │  (yfinance)     │        │   (Tavily)      │
           └─────────────────┘        └─────────────────┘        └─────────────────┘
```

**Key Components:**

- **Query Router**: Uses LLM to classify query complexity and estimate required tools
- **ReAct Agent**: For simple queries (1 tool, straightforward), uses dynamic reasoning
- **Query Planner**: Decomposes complex queries into structured `AnalysisStep` objects
- **Step Executor**: Executes plan steps sequentially, respecting dependencies
- **Synthesizer**: Combines multi-step results into comprehensive final response

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
│   │   └── sec_graph.py       # LangGraph planning workflow
│   ├── planner.py             # Query complexity classification & planning
│   ├── prompts.py             # System prompts for agents
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
