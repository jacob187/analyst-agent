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

    // Add more paragraphs as needed:
    // `Your additional paragraph here...`,
  ],
};

export const architectureContent = {
  title: "Architecture",

  systemOverview: {
    title: "System Overview",
    description: `The Analyst Agent is a full-stack application that combines real-time SEC filing data
with market analysis, powered by an AI agent that can reason about financial information.`,
  },

  agentFlow: {
    title: "LangGraph ReAct Agent Flow",
    description: `The agent uses the ReAct (Reasoning + Acting) pattern to process queries. It reasons about
which tools to use, executes them, observes results, and synthesizes a response.`,
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
│                           ANALYST AGENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         WebSocket          ┌──────────────────────────┐  │
│   │   Frontend   │ ◄─────────────────────────►│     FastAPI Backend      │  │
│   │   (Svelte)   │                            │    /ws/chat/{ticker}     │  │
│   └──────────────┘                            └────────────┬─────────────┘  │
│                                                            │                 │
│                                                            ▼                 │
│                                               ┌──────────────────────────┐  │
│                                               │    LangGraph ReAct Agent │  │
│                                               │    (create_react_agent)  │  │
│                                               └────────────┬─────────────┘  │
│                                                            │                 │
│                            ┌───────────────────────────────┼────────────┐   │
│                            │                               │            │   │
│                            ▼                               ▼            ▼   │
│               ┌────────────────────────┐    ┌─────────────────────────────┐ │
│               │     SEC Tools (8)      │    │   Yahoo Finance Tools (3)   │ │
│               │                        │    │                             │ │
│               │  • Risk Factors        │    │  • Stock Price History      │ │
│               │  • MD&A Analysis       │    │  • Technical Analysis       │ │
│               │  • Balance Sheets      │    │  • Stock Info               │ │
│               │  • Complete 10-K       │    │                             │ │
│               │  • All Summaries       │    │                             │ │
│               └───────────┬────────────┘    └──────────────┬──────────────┘ │
│                           │                                │                 │
│                           ▼                                ▼                 │
│               ┌────────────────────────┐    ┌─────────────────────────────┐ │
│               │      SEC EDGAR API     │    │      Yahoo Finance API      │ │
│               │   (edgartools library) │    │    (yfinance library)       │ │
│               └────────────────────────┘    └─────────────────────────────┘ │
│                                                                              │
│                           ┌────────────────────────────────┐                 │
│                           │      Google Gemini LLM         │                 │
│                           │  (langchain_google_genai)      │                 │
│                           └────────────────────────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘`,

  reactFlow: `
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LangGraph ReAct Agent Flow                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    User Query                                                                │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────┐    ┌──────────────────────────────────────────────────────┐   │
│   │  START  │───►│                   ReAct Agent                         │   │
│   └─────────┘    │                                                       │   │
│                  │   ┌─────────────────────────────────────────────────┐ │   │
│                  │   │  REASON: Analyze query, decide which tool(s)    │ │   │
│                  │   └─────────────────────────────────────────────────┘ │   │
│                  │                         │                             │   │
│                  │                         ▼                             │   │
│                  │   ┌─────────────────────────────────────────────────┐ │   │
│                  │   │  ACT: Call selected tool(s)                     │ │   │
│                  │   │  • SEC tools for filings/risks/financials       │ │   │
│                  │   │  • Yahoo tools for prices/technical analysis    │ │   │
│                  │   └─────────────────────────────────────────────────┘ │   │
│                  │                         │                             │   │
│                  │                         ▼                             │   │
│                  │   ┌─────────────────────────────────────────────────┐ │   │
│                  │   │  OBSERVE: Process tool output                   │ │   │
│                  │   └─────────────────────────────────────────────────┘ │   │
│                  │                         │                             │   │
│                  │              ┌──────────┴──────────┐                  │   │
│                  │              ▼                     ▼                  │   │
│                  │        Need more info?       Have enough?             │   │
│                  │              │                     │                  │   │
│                  │              └───────┐   ┌────────┘                   │   │
│                  │                      │   │                            │   │
│                  │                      ▼   ▼                            │   │
│                  │   ┌─────────────────────────────────────────────────┐ │   │
│                  │   │  SYNTHESIZE: Generate final response            │ │   │
│                  │   └─────────────────────────────────────────────────┘ │   │
│                  └──────────────────────────┬───────────────────────────┘   │
│                                             │                                │
│                                             ▼                                │
│                                        ┌─────────┐                           │
│                                        │   END   │                           │
│                                        └─────────┘                           │
│                                             │                                │
│                                             ▼                                │
│                                      Response to User                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘`,
};
