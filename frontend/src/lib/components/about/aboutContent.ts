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
        "Raw and AI-analyzed 10-K/10-Q sections — risk factors, MD&A, balance sheets, business overviews, and more — sourced directly from SEC EDGAR.",
    },
    {
      category: "Market Data & Technicals",
      description:
        "Price history, financial metrics, and technical indicators (RSI, MACD, Bollinger Bands, moving averages) with interactive charting.",
    },
    {
      category: "Web Research",
      description:
        "Company news, competitor analysis, and industry trends via optional Tavily integration.",
    },
  ],
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
│                                      │            │  (Parallel layers)  │    │
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
│               │  SEC Tools        │  │  Yahoo Finance  │  │  Tavily      │ │
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
│  • Automatic complexity detection                                           │
│  • Dynamic routing to optimal execution path                                │
│  • Structured planning for multi-step analysis                              │
│  • Parallel execution of independent steps (~2-4x faster)                   │
│  • Dependency-aware layer ordering                                          │
│  • Intelligent result synthesis                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘`,
};
