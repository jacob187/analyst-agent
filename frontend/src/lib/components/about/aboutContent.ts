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
    financial analysis. This hybrid approach optimizes both performance and result quality.`,
  ],
};

export const architectureContent = {
  title: "Architecture",

  systemOverview: {
    title: "System Overview",
    description: `The Analyst Agent is a full-stack application that combines real-time SEC filing data
with market analysis, powered by a hybrid AI agent architecture. The system intelligently routes queries
based on complexity - using dynamic ReAct for simple questions and structured multi-step planning for
complex financial analysis.`,
  },

  agentFlow: {
    title: "Intelligent Query Planning Flow",
    description: `The agent uses query complexity classification to determine the optimal execution path.
Simple queries are handled by a ReAct agent with dynamic tool selection. Complex queries are decomposed
into structured execution plans with explicit steps, dependencies, and result synthesis.`,
  },
};

export const toolsContent = {
  title: "Available Tools",

  secTools: {
    category: "SEC Filing Tools",
    description: "Data from SEC EDGAR via edgartools library",
    tools: [
      {
        name: "get_raw_risk_factors",
        desc: "Complete raw text of Risk Factors section (Item 1A) from 10-K filing",
      },
      {
        name: "get_risk_factors_summary",
        desc: "LLM-analyzed risk factors with sentiment score (1-10), key risks, and risk categories",
      },
      {
        name: "get_raw_management_discussion",
        desc: "Raw MD&A (Management Discussion & Analysis) text from Item 7",
      },
      {
        name: "get_mda_summary",
        desc: "Analyzed MD&A with sentiment score, future outlook, financial highlights, and key points",
      },
      {
        name: "get_raw_balance_sheets",
        desc: "Balance sheet data as JSON from both 10-K and 10-Q filings",
      },
      {
        name: "get_balance_sheet_summary",
        desc: "Financial health analysis with key metrics, liquidity/solvency analysis, and red flags",
      },
      {
        name: "get_complete_10k_text",
        desc: "List of all available 10-K sections that can be retrieved as raw text",
      },
      {
        name: "get_all_summaries",
        desc: "Comprehensive overview combining risk factors, MD&A, and financial analysis",
      },
    ],
  },

  yahooTools: {
    category: "Yahoo Finance Tools",
    description: "Real-time market data via yfinance library",
    tools: [
      {
        name: "get_stock_price_history",
        desc: "Last 10 trading days with OHLC prices, volume, and period change percentage",
      },
      {
        name: "get_technical_analysis",
        desc: "RSI (14-day), MACD with signal line, Bollinger Bands, Moving Averages (5/10/20/50/200-day), and volatility metrics",
      },
      {
        name: "get_stock_info",
        desc: "Current price, P/E ratio, market cap, 52-week high/low, average volume, dividend yield, and beta",
      },
    ],
  },

  tavilyTools: {
    category: "Tavily Research Tools",
    description: "Web research and news via Tavily API (optional)",
    tools: [
      {
        name: "web_search",
        desc: "General company information search across the web",
      },
      {
        name: "deep_research",
        desc: "Multi-source comprehensive research on company topics",
      },
      {
        name: "get_company_news",
        desc: "Latest news and developments for the company",
      },
      {
        name: "analyze_competitors",
        desc: "Market positioning and competitor analysis",
      },
      {
        name: "get_industry_trends",
        desc: "Industry outlook and trend forecasts",
      },
    ],
  },
};

export const techStackContent = {
  title: "Technology Stack",

  items: [
    { label: "Frontend", value: "Svelte 5 + TypeScript + Vite" },
    { label: "Backend", value: "FastAPI + WebSocket" },
    { label: "AI Framework", value: "LangGraph + LangChain" },
    { label: "LLM", value: "Google Gemini" },
    { label: "SEC Data", value: "edgartools (EDGAR API)" },
    { label: "Market Data", value: "yfinance (Yahoo Finance)" },
    { label: "Web Research", value: "Tavily API (optional)" },
  ],
};

// ASCII Diagrams - Edit these if you want to change the visual diagrams
export const diagrams = {
  systemArchitecture: `
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ANALYST AGENT ARCHITECTURE (v2)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         WebSocket          ┌──────────────────────────┐  │
│   │   Frontend   │ ◄─────────────────────────►│     FastAPI Backend      │  │
│   │   (Svelte)   │                            │    /ws/chat/{ticker}     │  │
│   └──────────────┘                            └────────────┬─────────────┘  │
│                                                            │                 │
│                                                            ▼                 │
│                                               ┌──────────────────────────┐  │
│                                               │ LangGraph Planning Agent │  │
│                                               │  (StateGraph workflow)   │  │
│                                               └────────────┬─────────────┘  │
│                                                            │                 │
│                                  ┌─────────────────────────┴─────────────┐  │
│                                  │      Query Complexity Router          │  │
│                                  │  (LLM-based classification)           │  │
│                                  └──────────┬───────────────┬────────────┘  │
│                                             │               │                │
│                                  SIMPLE     │               │    COMPLEX     │
│                                             │               │                │
│                                             ▼               ▼                │
│                              ┌────────────────┐   ┌────────────────────┐    │
│                              │  ReAct Agent   │   │  Query Planner     │    │
│                              │  (Dynamic)     │   │  (Structured)      │    │
│                              └───────┬────────┘   └─────────┬──────────┘    │
│                                      │                      │                │
│                                      │                      ▼                │
│                                      │            ┌─────────────────────┐    │
│                                      │            │  Step Executor      │    │
│                                      │            │  (Loop until done)  │    │
│                                      │            └─────────┬───────────┘    │
│                                      │                      │                │
│                                      │                      ▼                │
│                                      │            ┌─────────────────────┐    │
│                                      │            │   Synthesizer       │    │
│                                      │            │  (Combine results)  │    │
│                                      │            └─────────┬───────────┘    │
│                                      │                      │                │
│                                      └──────────┬───────────┘                │
│                                                 │                            │
│                            ┌────────────────────┼────────────────────┐       │
│                            │                    │                    │       │
│                            ▼                    ▼                    ▼       │
│               ┌────────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│               │   SEC Tools (8)    │  │  Yahoo (3)      │  │ Tavily (5)   │ │
│               │                    │  │                 │  │  (optional)  │ │
│               │ • Risk Factors     │  │ • Price History │  │ • News       │ │
│               │ • MD&A             │  │ • Technicals    │  │ • Research   │ │
│               │ • Balance Sheets   │  │ • Stock Info    │  │ • Industry   │ │
│               └────────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                                              │
│                           ┌────────────────────────────────┐                 │
│                           │      Google Gemini LLM         │                 │
│                           │  (Powers classification,       │                 │
│                           │   planning, and synthesis)     │                 │
│                           └────────────────────────────────┘                 │
│                                                                              │
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
│         │  │  Reason    │  │        │  plan with steps:   │                 │
│         │  │  Select    │  │        │  • Step 1: Action   │                 │
│         │  │  Tool      │  │        │  • Step 2: Depends  │                 │
│         │  └─────┬──────┘  │        │    on Step 1        │                 │
│         │        │         │        │  • Step N: Finalize │                 │
│         │        ▼         │        └──────────┬──────────┘                 │
│         │  ┌────────────┐  │                   │                             │
│         │  │  Execute   │  │                   ▼                             │
│         │  │  Tool      │  │        ┌─────────────────────┐                 │
│         │  └─────┬──────┘  │        │  Step Executor      │◄─┐              │
│         │        │         │        │                     │  │              │
│         │        ▼         │        │  Execute current    │  │              │
│         │  ┌────────────┐  │        │  step with tool     │  │              │
│         │  │  Response  │  │        └──────────┬──────────┘  │              │
│         │  └────────────┘  │                   │              │              │
│         └─────────┬────────┘                   ▼              │              │
│                   │                    ┌───────────────┐      │              │
│                   │                    │  More steps?  │──Yes─┘              │
│                   │                    └───────┬───────┘                     │
│                   │                            │ No                          │
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
│  • Automatic complexity detection                                           │
│  • Dynamic routing to optimal execution path                                │
│  • Structured planning for multi-step analysis                              │
│  • Dependency-aware step execution                                          │
│  • Intelligent result synthesis                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘`,
};
