/**
 * ABOUT PAGE CONTENT
 *
 * Edit this file to customize all text on the About page.
 * The structure below maps to each section of the page.
 */

export const storyContent = {
  title: "The Story",

  // EDIT YOUR STORY HERE:
  paragraphs: [
    `This project was born from the challenge of parsing SEC N-PX filings
    to understand institutional investor voting patterns.`,

    `What started as a data extraction problem evolved into a comprehensive
    AI-powered financial analysis platform.`,

    `The architecture evolved further from a simple ReAct agent to an intelligent
    planning system that routes queries based on complexity - using dynamic tool
    selection for simple questions and structured multi-step execution for complex
    financial analysis. Independent analysis steps run in parallel to minimize
    response time. This hybrid approach optimizes both performance and result quality.`,
  ],
};

export const architectureContent = {
  title: "Architecture",

  systemOverview: {
    title: "System Overview",
    description: `The Analyst Agent is a full-stack application that combines real-time SEC filing data
with market analysis, powered by a hybrid AI agent architecture. The system intelligently routes queries
based on complexity - using dynamic ReAct for simple questions and structured multi-step planning for
complex financial analysis. Independent steps are executed in parallel to minimize latency.`,
  },

  agentFlow: {
    title: "Intelligent Query Planning Flow",
    description: `The agent uses query complexity classification to determine the optimal execution path.
Simple queries are handled by a ReAct agent with dynamic tool selection. Complex queries are decomposed
into structured execution plans where steps are grouped into dependency layers - independent steps within
a layer run concurrently, while layers execute in sequence to respect data dependencies.`,
  },
};

export const capabilitiesContent = {
  title: "Capabilities",

  categories: [
    {
      category: "SEC Filing Analysis",
      description:
        "11 tools powered by edgartools. Raw and AI-analyzed 10-K/10-Q sections — risk factors, MD&A, balance sheets, business overview, cybersecurity disclosure, legal proceedings, and a concurrent all-summaries tool. Prefers 10-Q for recency with 10-K fallback.",
    },
    {
      category: "Market Data & Technicals",
      description:
        "7 tools via Yahoo Finance. Price history, stock info (P/E, market cap, beta), financial metrics (revenue/income growth, debt ratios), core technicals (RSI, MACD, Bollinger Bands, 5/10/20/50/200-day MAs), advanced technicals (ADX, ATR, Stochastic, Volume Profile, Fibonacci), chart pattern detection (Head & Shoulders, Double Top/Bottom, Golden/Death Cross), and multi-timeframe analysis with conflict detection.",
    },
    {
      category: "Market Overview",
      description:
        "2 tools for broad market context. Current levels and daily changes for S&P 500, Nasdaq, Dow Jones, and VIX. Macro indicators including 10-Year and 5-Year Treasury yields, VIX, and US Dollar Index with 1-month changes.",
    },
    {
      category: "Web Research",
      description:
        "5 optional tools via Tavily. General web search, deep research (async polling), company news, competitor analysis, and industry trend forecasting. Requires a Tavily API key.",
    },
    {
      category: "Briefing History",
      description:
        "2 tools for daily briefing data. Retrieve recent briefing analyses for a ticker (outlook trends, technical signals, news summaries) or the latest full morning briefing (market regime, positioning, alerts).",
    },
  ],
};

export const techStackContent = {
  title: "Technology Stack",

  items: [
    { label: "Frontend", value: "Svelte 5 + TypeScript + Vite" },
    { label: "Backend", value: "FastAPI + WebSocket" },
    { label: "AI Framework", value: "LangGraph + LangChain" },
    { label: "LLM", value: "Multi-provider (Google Gemini, OpenAI, Anthropic)" },
    { label: "SEC Data", value: "edgartools (EDGAR API)" },
    { label: "Market Data", value: "yfinance (Yahoo Finance)" },
    { label: "Web Research", value: "Tavily API (optional)" },
  ],
};

// ASCII Diagrams - Edit these if you want to change the visual diagrams
export const diagrams = {
  systemArchitecture: `
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANALYST AGENT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐          WebSocket           ┌─────────────────────────┐  │
│  │   Frontend   │ ◄──────────────────────────► │    FastAPI Backend      │  │
│  │  (Svelte 5)  │                              │   /ws/chat/{ticker}     │  │
│  └──────────────┘                              └───────────┬─────────────┘  │
│                                                            │                │
│                                                            ▼                │
│                                               ┌─────────────────────────┐   │
│                                               │ LangGraph StateGraph    │   │
│                                               │  (Planning Agent)       │   │
│                                               └───────────┬─────────────┘   │
│                                                           │                 │
│                                 ┌─────────────────────────┴──────────────┐  │
│                                 │      Query Complexity Router           │  │
│                                 │  (LLM-based classification)            │  │
│                                 └─────────┬──────────────────┬───────────┘  │
│                                           │                  │              │
│                                SIMPLE     │                  │   COMPLEX    │
│                                           ▼                  ▼              │
│                            ┌────────────────┐   ┌────────────────────┐      │
│                            │  ReAct Agent   │   │  Query Planner     │      │
│                            │  (Dynamic)     │   │  (Structured)      │      │
│                            └───────┬────────┘   └─────────┬──────────┘      │
│                                    │                      │                 │
│                                    │                      ▼                 │
│                                    │            ┌────────────────────┐      │
│                                    │            │  Step Executor     │      │
│                                    │            │  (Parallel layers) │      │
│                                    │            └─────────┬──────────┘      │
│                                    │                      │                 │
│                                    │                      ▼                 │
│                                    │            ┌────────────────────┐      │
│                                    │            │   Synthesizer      │      │
│                                    │            │  (Thinking LLM)    │      │
│                                    │            └─────────┬──────────┘      │
│                                    │                      │                 │
│                                    └──────────┬───────────┘                 │
│                                               │                             │
│          ┌──────────────┬─────────────┬───────┼───────────┬──────────┐      │
│          ▼              ▼             ▼       ▼           ▼          ▼      │
│  ┌──────────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐    │
│  │  SEC Tools   │ │  Stock    │ │ Market  │ │ Tavily  │ │  Briefing  │    │
│  │  (11 tools)  │ │  Tools    │ │ Tools   │ │ Tools   │ │  Tools     │    │
│  │              │ │ (7 tools) │ │(2 tools)│ │(5 tools)│ │ (2 tools)  │    │
│  │ • Filings   │ │ • Price   │ │ • Index │ │ • News  │ │ • History  │    │
│  │ • Analysis  │ │ • Technls │ │ • Macro │ │ • Deep  │ │ • Latest   │    │
│  │ • Balance   │ │ • Patterns│ │ • Yields│ │ • Comps │ │   briefing │    │
│  └──────────────┘ └───────────┘ └─────────┘ └─────────┘ └────────────┘    │
│                                                                             │
│                    ┌─────────────────────────────────────┐                  │
│                    │  Multi-Provider LLM (8 models)      │                  │
│                    │  Google Gemini │ OpenAI │ Anthropic  │                  │
│                    │  (classification, planning,          │                  │
│                    │   tool calls, and synthesis)         │                  │
│                    └─────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘`,

  reactFlow: `
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Intelligent Query Planning Workflow                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              User Query                                      │
│                                  │                                           │
│                                  ▼                                           │
│                          ┌───────────────┐                                   │
│                          │  Query Router │                                   │
│                          │  (Classify)   │                                   │
│                          └───────┬───────┘                                   │
│                                  │                                           │
│                    ┌─────────────┴─────────────┐                             │
│                    │                           │                             │
│            SIMPLE (≤1 tool)           COMPLEX (>1 tool)                      │
│                    │                           │                             │
│                    ▼                           ▼                             │
│         ┌──────────────────┐        ┌─────────────────────┐                 │
│         │   ReAct Agent    │        │   Query Planner     │                 │
│         │                  │        │                     │                 │
│         │  ┌────────────┐  │        │  Creates structured │                 │
│         │  │  Reason    │  │        │  plan with steps    │                 │
│         │  │  Select    │  │        │  and dependencies:  │                 │
│         │  │  Tool      │  │        │  • Layer 0: A, B    │                 │
│         │  └─────┬──────┘  │        │    (parallel)       │                 │
│         │        │         │        │  • Layer 1: C       │                 │
│         │        ▼         │        │    (depends on A,B) │                 │
│         │  ┌────────────┐  │        └──────────┬──────────┘                 │
│         │  │  Execute   │  │                   │                             │
│         │  │  Tool      │  │                   ▼                             │
│         │  └─────┬──────┘  │        ┌─────────────────────┐                 │
│         │        │         │        │  Step Executor      │                 │
│         │        ▼         │        │                     │                 │
│         │  ┌────────────┐  │        │  Layer 0: ─┬─ A     │                 │
│         │  │  Response  │  │        │            └─ B     │                 │
│         │  └────────────┘  │        │  Layer 1: ── C      │                 │
│         └─────────┬────────┘        │  (parallel within   │                 │
│                   │                 │   each layer)       │                 │
│                   │                 └──────────┬──────────┘                 │
│                   │                            │                            │
│                   │                            ▼                             │
│                   │                   ┌─────────────────┐                    │
│                   │                   │  Synthesizer    │                    │
│                   │                   │                 │                    │
│                   │                   │  Combine all    │                    │
│                   │                   │  step results   │                    │
│                   │                   │  into coherent  │                    │
│                   │                   │  response       │                    │
│                   │                   └────────┬────────┘                    │
│                   │                            │                             │
│                   └────────────┬───────────────┘                             │
│                                │                                             │
│                                ▼                                             │
│                         ┌─────────────┐                                      │
│                         │    END      │                                      │
│                         └──────┬──────┘                                      │
│                                │                                             │
│                                ▼                                             │
│                         Response to User                                     │
│                                                                              │
│  Key Features:                                                               │
│  • Automatic complexity detection (simple ≤1 tool, complex >1)              │
│  • Dynamic routing to optimal execution path                                │
│  • Structured planning with dependency-aware step ordering                  │
│  • Parallel execution of independent steps via ThreadPoolExecutor           │
│  • Real-time streaming (node progress, tool events, thinking, tokens)       │
│  • Multi-provider LLM support (Gemini, OpenAI, Anthropic — 8 models)       │
│  • Dual-LLM architecture: tool LLM + thinking-enabled synthesizer          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘`,
};
