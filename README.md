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

## API Documentation

Interactive API docs are available at `http://localhost:8000/docs` when running locally.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Edgar Tools](https://github.com/dgunning/edgartools) - SEC filings retrieval
- [Yahoo Finance](https://pypi.org/project/yfinance/) - Market data
- [LangChain](https://python.langchain.com/) - LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent framework
- [Tavily](https://tavily.com/) - Web research API
- [Svelte](https://svelte.dev/) - Frontend framework
