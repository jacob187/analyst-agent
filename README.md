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
- **LangChain** - LLM framework (Google Gemini, OpenAI, Anthropic)

### Data Sources
- **SEC EDGAR** - 10-K and 10-Q filings via edgartools
- **Yahoo Finance** - Market data and technical indicators
- **Tavily** - Web research and news (optional)

## Features

- **Interactive Chat Interface**: Real-time WebSocket communication with AI agent
- **Stock Charts**: Candlestick charts with technical indicators (RSI, MACD, Bollinger Bands, moving averages) via TradingView Lightweight Charts
- **Watchlist & Briefings**: Track tickers and generate AI-powered morning briefings with market regime analysis
- **Intelligent Query Routing**: Automatically classifies queries and routes to optimal execution path
- **Multi-Step Planning**: Complex queries are decomposed into structured execution plans
- **SEC Filings Analysis**: Extracts and analyzes MD&A, Risk Factors, and Balance Sheets
- **Technical Analysis**: RSI, MACD, Bollinger Bands, moving averages, pattern detection, and more
- **Web Research**: Company news, competitor analysis, and industry trends (with Tavily)
- **Hybrid Agent Architecture**:
  - Simple queries → ReAct agent (dynamic tool selection)
  - Complex queries → Planner → Step-by-step execution → Synthesis
- **Smart Caching**: 3-tier caching system minimizes redundant API calls
- **Privacy-First**: API keys stay in your browser — the server never stores credentials

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

Starts both the backend (port 8000) and frontend (port 5173). If you skip `install.sh`, `start.sh` will auto-install dependencies on first run.

### Configure API Keys

**Bring your own API key.** Supports Google Gemini, OpenAI, and Anthropic Claude. You only need one provider — pick whichever you prefer.

**Option 1: Settings UI (recommended for hosted demo)**

Open http://localhost:5173 and select your provider and model in **Settings**:

| Provider | Get a key | Notes |
|----------|-----------|-------|
| Google Gemini | [Google AI Studio](https://aistudio.google.com/app/apikey) | Free tier available |
| OpenAI | [OpenAI Platform](https://platform.openai.com/api-keys) | GPT-4o Mini is cost-effective |
| Anthropic | [Anthropic Console](https://console.anthropic.com/settings/keys) | Claude Sonnet is recommended |

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
