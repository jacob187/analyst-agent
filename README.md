# Stock Analyst Agent

A full-stack AI-powered financial analysis application that combines SEC filings analysis, technical indicators, and real-time web research through an interactive company dashboard and chat interface.

## Tech Stack

### Frontend
- **Next.js 16** - React framework (App Router)
- **React 19** - UI library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS v4** - Utility-first styling
- **shadcn/ui + Radix** - Accessible component primitives
- **TradingView Lightweight Charts** - Interactive candlestick charts

### Backend
- **FastAPI** - Python async web framework
- **WebSocket** - Real-time bidirectional chat
- **SSE (Server-Sent Events)** - Streaming filing analysis progress
- **LangGraph** - Graph-based AI agent orchestration
- **LangChain** - LLM framework (Google Gemini, OpenAI, Anthropic)
- **SQLite (aiosqlite)** - Sessions, messages, watchlist, filing cache

### Data Sources
- **SEC EDGAR** - 10-K, 10-Q, and 8-K filings via edgartools (including 20-F for foreign filers)
- **Yahoo Finance** - Market data, financials, and technical indicators
- **Tavily** - Web research and news (optional)

## Features

- **Company Dashboard**: Full-page view per ticker with Overview and Filings tabs
  - **Overview**: Live quote, key metrics (P/E, market cap, 52-wk range, beta, dividend yield), technical snapshot (RSI, MACD, ADX, Bollinger Bands), detected chart patterns, market regime, earnings calendar, and company profile
  - **Filings**: LLM-analyzed SEC sections streamed via SSE — 10-K risk factors, 10-K MD&A, balance sheet, 10-Q risk factors, 10-Q MD&A, and 8-K earnings analysis. Results cached in SQLite by accession number and auto-invalidated on new filings
- **Interactive Chat**: Real-time WebSocket communication with AI agent, integrated into the company dashboard
- **Stock Charts**: Candlestick charts with RSI, MACD, Bollinger Bands, and moving averages via TradingView Lightweight Charts
- **Watchlist & Briefings**: Track up to 10 tickers and generate AI-powered morning briefings with market regime analysis, positioning signals, and per-ticker outlook
- **Companies Page**: Grid of all previously analyzed tickers for quick navigation
- **Session History**: Browse and review past chat sessions grouped by ticker
- **Intelligent Query Routing**: Classifies query complexity and routes to the optimal execution path
- **Multi-Step Planning**: Complex queries are decomposed into structured execution plans
- **SEC Filings Analysis**: Extracts and analyzes MD&A, Risk Factors, Balance Sheets from 10-K, 10-Q, and 8-K filings
- **Advanced Technical Analysis**: RSI, MACD, Bollinger Bands, ADX, ATR (with stop-loss suggestion), Stochastic, Volume Profile (POC/value area), Fibonacci retracement, and pattern detection (Head & Shoulders, Double Top/Bottom, Golden/Death Cross, RSI divergences)
- **Multi-Timeframe Analysis**: Daily, weekly, and hourly analysis with conflict detection and bias recommendation
- **Web Research**: Company news, competitor analysis, and industry trends (with Tavily)
- **Hybrid Agent Architecture**:
  - Simple queries → ReAct agent (dynamic tool selection)
  - Complex queries → Planner → Step-by-step execution → Synthesis
- **Smart Caching**: In-memory caching (SEC retrievers, LLM analysis, research results) plus SQLite-backed filing analysis cache
- **Privacy-First**: API keys stay in your browser's localStorage — the server never stores credentials

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page — enter a ticker to start analysis |
| `/company/[ticker]` | Company dashboard with Overview and Filings tabs + integrated chat |
| `/companies` | Grid of all previously analyzed tickers |
| `/watchlist` | Manage tracked tickers and generate AI morning briefings |
| `/history` | Browse past chat sessions grouped by ticker |
| `/session/[id]` | Read-only archived session viewer |
| `/settings` | API key configuration (stored in browser localStorage) |
| `/about` | Architecture explainer with agent graph diagram |

## Supported Models

You only need one provider — pick whichever you prefer.

| Model | Provider | Context | Thinking |
|-------|----------|---------|----------|
| **Gemini 3 Flash** (default) | Google | 1M | Yes |
| Gemini 3.1 Pro | Google | 1M | Yes |
| GPT-5.4 | OpenAI | 1.05M | No |
| GPT-4.1 Mini | OpenAI | 1M | No |
| o4 Mini | OpenAI | 200K | Yes |
| o3 | OpenAI | 200K | Yes |
| Claude Sonnet 4.6 | Anthropic | 200K | Yes |
| Claude Opus 4.6 | Anthropic | 200K | Yes |

## Quick Start

### Prerequisites

You need **Python 3.12+**, **Node.js 20+**, and **uv** installed before running the app.

<details>
<summary><strong>Install uv</strong> (Python package manager)</summary>

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew
brew install uv
```

See the [uv docs](https://docs.astral.sh/uv/getting-started/installation/) for more options.

</details>

<details>
<summary><strong>Install Node.js 20+</strong></summary>

```bash
# macOS (Homebrew)
brew install node

# Or use a version manager like nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install 20
```

See [nodejs.org](https://nodejs.org/) for installers and other methods.

</details>

### Install

```bash
./install.sh
```

This installs Python and frontend dependencies, creates a `.env` file from `.env.example`, and sets up the data directory.

### Run the Application

```bash
./start.sh
```

Starts both the backend (port 8000) and frontend (port 3000). If you skip `install.sh`, `start.sh` will auto-install dependencies on first run.

### Configure API Keys

**Bring your own API key.** Supports Google Gemini, OpenAI, and Anthropic Claude. You only need one provider — pick whichever you prefer.

**Option 1: Settings UI (recommended for hosted demo)**

Open http://localhost:3000/settings and select your provider and model:

| Provider | Get a key | Notes |
|----------|-----------|-------|
| Google Gemini | [Google AI Studio](https://aistudio.google.com/app/apikey) | Free tier available; Gemini 3 Flash is the default model |
| OpenAI | [OpenAI Platform](https://platform.openai.com/api-keys) | GPT-4.1 Mini is cost-effective |
| Anthropic | [Anthropic Console](https://console.anthropic.com/settings/keys) | Claude Sonnet 4.6 recommended |

Also required:
- **SEC Header** — Your name and email (SEC requires identification per their [fair access policy](https://www.sec.gov/os/webmaster-faq#code-support))
- **Tavily API Key** — Optional, enables web research ([sign up](https://app.tavily.com/sign-in))

Keys are stored in your browser's `localStorage` only — they are never saved on the server.

**Option 2: `.env` file (recommended for local development)**

Create a `.env` file in the project root:

```bash
# At least one LLM provider key is required
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

SEC_HEADER=Your Name your.email@example.com
TAVILY_API_KEY=tvly-...  # optional
```

The backend loads `.env` at startup and uses these as fallbacks when no key is provided via the frontend.

> **Note:** `.env` is gitignored — your keys will never be committed.

## API Documentation

Interactive API docs are available at `http://localhost:8000/docs` when running locally.

## Architecture

### System Architecture

```
┌─────────────────┐  WebSocket / SSE   ┌─────────────────┐
│    Frontend     │◄──────────────────►│    FastAPI      │
│   (Next.js)    │     REST            │    Backend      │
└─────────────────┘                    └────────┬────────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              │                 │                 │
                     ┌────────▼────────┐ ┌──────▼──────┐ ┌───────▼───────┐
                     │  LangGraph      │ │  Company    │ │  Watchlist    │
                     │  Planning Agent │ │  Profiles   │ │  & Briefings  │
                     └────────┬────────┘ └─────────────┘ └───────────────┘
                              │
          ┌───────────────────┼───────────────────┬───────────────────┐
          │                   │                   │                   │
 ┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐ ┌────────▼────────┐
 │   SEC Tools     │ │  Stock Tools  │ │  Market Tools   │ │ Research Tools  │
 │  (edgartools)   │ │  (yfinance)   │ │  (macro data)   │ │   (Tavily)      │
 └─────────────────┘ └───────────────┘ └─────────────────┘ └─────────────────┘
```

### Query Planning Workflow

The agent uses a hybrid architecture that routes queries based on complexity:

```
User Query → Router → Simple  → ReAct Agent → Response
                    → Complex → Planner → Step Executor (loop) → Synthesizer → Response
```

- **Query Router**: LLM classifies query complexity and estimates required tools
- **ReAct Agent**: For simple queries — dynamic tool selection in a single reasoning loop
- **Query Planner**: Decomposes complex queries into structured `AnalysisStep` objects
- **Step Executor**: Executes plan steps sequentially, respecting dependencies
- **Synthesizer**: Combines multi-step results into a comprehensive final response

### Backend Routes

| Route | Transport | Description |
|-------|-----------|-------------|
| `/ws/chat/{ticker}` | WebSocket | Real-time chat with AI agent |
| `/api/company/{ticker}/profile` | REST | Aggregated company data (quote, technicals, patterns, regime) |
| `/api/company/{ticker}/filings` | REST | Batch LLM filing analysis |
| `/api/company/{ticker}/filings/stream` | SSE | Streaming filing analysis with per-section progress |
| `/stock/{ticker}/chart` | REST | OHLCV chart data |
| `/models` | REST | Available model list |
| `/watchlist` | REST | CRUD for tracked tickers (max 10) |
| `/watchlist/briefing` | REST | Generate AI morning briefing |
| `/watchlist/briefing/history` | REST | Past briefings |
| `/sessions` | REST | Session CRUD |
| `/health` | REST | Health check |

### Database (SQLite)

| Table | Purpose |
|-------|---------|
| `sessions` | Chat sessions with ticker, model, and compressed summary |
| `messages` | Chat messages per session |
| `companies` | Central company entity (ticker, name, sector) |
| `watchlist` | Tracked tickers (max 10) |
| `briefings` | Daily briefing results (regime, positioning, alerts) |
| `briefing_tickers` | Per-ticker data within a briefing |
| `filings_cache` | SEC filing metadata (form type, accession number, download status) |
| `filing_analyses` | LLM-generated filing analyses cached by accession number |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Edgar Tools](https://github.com/dgunning/edgartools) - SEC filings retrieval
- [Yahoo Finance](https://pypi.org/project/yfinance/) - Market data
- [LangChain](https://python.langchain.com/) - LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent framework
- [Tavily](https://tavily.com/) - Web research API
- [Next.js](https://nextjs.org/) - Frontend framework
- [shadcn/ui](https://ui.shadcn.com/) - UI components
- [TradingView Lightweight Charts](https://tradingview.github.io/lightweight-charts/) - Charting library
