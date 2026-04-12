import {
  FileText,
  BarChart2,
  Globe,
  Search,
  Newspaper,
  LayoutDashboard,
  LineChart,
  MessageSquare,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const GRAPH = `
START
  └─▶ [ROUTER]
       ├─▶ "simple"  ──▶ [REACT AGENT] ──▶ END
       └─▶ "complex" ──▶ [PLANNER]
                               └─▶ [STEP EXECUTOR] ⟵──┐
                                         └─▶ (loop)  ──┘
                                         └─▶ [SYNTHESIZER] ──▶ END
`.trim();

const CAPABILITIES = [
  {
    icon: FileText,
    category: "SEC Filing Analysis",
    count: "15 tools",
    description:
      "Powered by edgartools. Raw and AI-analyzed 10-K/10-Q sections — risk factors, MD&A, balance sheets, full 10-K text, business overview, cybersecurity disclosure, legal proceedings, and a concurrent all-summaries tool. 8-K support with earnings release analysis (beats/misses, guidance) and material event analysis. Supports 20-F for foreign filers.",
  },
  {
    icon: BarChart2,
    category: "Market Data & Technicals",
    count: "7 tools",
    description:
      "Via Yahoo Finance. Price history, stock info (P/E, market cap, beta), financial metrics (revenue/income growth, debt ratios), core technicals (RSI, MACD, Bollinger Bands, 5/10/20/50/200-day MAs), advanced technicals (ADX, ATR, Stochastic, Volume Profile, Fibonacci), chart pattern detection, and multi-timeframe analysis with conflict detection.",
  },
  {
    icon: Globe,
    category: "Market Overview",
    count: "2 tools",
    description:
      "Current levels and daily changes for S&P 500, Nasdaq, Dow Jones, and VIX. Macro indicators including 10-Year and 5-Year Treasury yields, VIX, and US Dollar Index with 1-month changes.",
  },
  {
    icon: Search,
    category: "Web Research",
    count: "5 tools",
    description:
      "Optional tools via Tavily. General web search, deep research (async polling), company news, competitor analysis, and industry trend forecasting. Requires a Tavily API key.",
  },
  {
    icon: Newspaper,
    category: "Briefing History",
    count: "2 tools",
    description:
      "Retrieve recent briefing analyses for a ticker (outlook trends, technical signals, news summaries) or the latest full morning briefing (market regime, positioning, alerts).",
  },
];

const DASHBOARD_FEATURES = [
  {
    icon: LayoutDashboard,
    label: "Instant Overview",
    description:
      "Selecting a ticker loads a metrics grid, technical snapshot, chart patterns, and market regime badge instantly — no LLM call, no prompt. Backed by a 5-minute server-side cache.",
  },
  {
    icon: FileText,
    label: "LLM-Analyzed Filings",
    description:
      "The Filings tab streams AI analysis of the latest 10-K (or 20-F), 10-Q, and 8-K earnings release via SSE. All analyses run concurrently and render incrementally as they complete. Cached in SQLite by accession number — auto-invalidates when a new filing appears on EDGAR.",
  },
  {
    icon: MessageSquare,
    label: "Integrated Chart + Chat",
    description:
      "The third tab embeds the full interactive chart (TradingView Lightweight Charts with RSI, MACD, Bollinger Bands) alongside the conversational agent — 31 tools available when Tavily is configured.",
  },
];

const STORY = [
  `This project was born from the challenge of parsing SEC N-PX filings
  to understand institutional investor voting patterns.`,
  `What started as a data extraction problem evolved into a comprehensive
  AI-powered financial analysis platform.`,
  `The architecture evolved from a simple ReAct agent to an intelligent
  planning system that routes queries based on complexity — using dynamic tool
  selection for simple questions and structured multi-step execution for complex
  financial analysis. Independent analysis steps run in parallel to minimize
  response time.`,
  `Most recently, the platform expanded beyond chat-only interaction with a
  Company Dashboard that proactively surfaces SEC filing analysis, technicals,
  and company metadata the moment a user selects a ticker — no prompt required.`,
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      {/* Header */}
      <div className="mb-10">
        <Badge variant="secondary" className="mb-4 rounded-full px-3 py-1 text-xs">
          Portfolio Project
        </Badge>
        <h1 className="font-display text-3xl font-bold sm:text-4xl">
          How it works
        </h1>
        <p className="mt-4 text-muted-foreground leading-relaxed">
          Analyst Agent is an AI-powered SEC filing analysis platform. Enter any
          public company ticker and ask natural language questions — the agent
          fetches SEC filings, runs LLM analysis, and streams results back in
          real time.
        </p>
      </div>

      {/* The Story */}
      <div className="mb-10">
        <h2 className="mb-4 font-display text-xl font-bold">The Story</h2>
        <div className="space-y-3">
          {STORY.map((paragraph, i) => (
            <p key={i} className="text-sm leading-relaxed text-muted-foreground">
              {paragraph}
            </p>
          ))}
        </div>
      </div>

      {/* Graph architecture */}
      <div className="mb-10 rounded-xl border border-border/60 bg-card p-5">
        <h2 className="mb-3 font-display text-base font-semibold">
          Graph Architecture
        </h2>
        <pre className="overflow-x-auto rounded-lg bg-muted px-4 py-3 text-xs leading-relaxed text-muted-foreground font-mono">
          {GRAPH}
        </pre>
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          The agent uses query complexity classification to determine the optimal
          execution path. Simple queries are handled by a ReAct agent with dynamic
          tool selection. Complex queries are decomposed into structured execution
          plans where independent steps run concurrently, and the synthesizer
          assembles the final response.
        </p>
      </div>

      {/* Caching */}
      <div className="mb-10 rounded-xl border border-border/60 bg-card p-5">
        <h2 className="mb-3 font-display text-base font-semibold">
          3-Tier Caching
        </h2>
        <div className="space-y-2">
          {[
            { tier: "1", what: "edgartools Company object", when: "Per process lifetime" },
            { tier: "2", what: "LLM analysis (Pydantic models)", when: "Per accession number (SQLite)" },
            { tier: "3", what: "Tavily web search results", when: "Per query hash (in-memory)" },
          ].map((c) => (
            <div key={c.tier} className="flex items-center gap-3 text-sm">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/20 text-[10px] font-bold text-primary">
                {c.tier}
              </span>
              <span className="font-medium">{c.what}</span>
              <span className="text-muted-foreground">· {c.when}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Capabilities */}
      <h2 className="mb-4 font-display text-xl font-bold">Capabilities</h2>
      <div className="mb-10 grid gap-4 sm:grid-cols-2">
        {CAPABILITIES.map((cap) => (
          <Card key={cap.category} className="border-border/60">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 font-display text-sm">
                <cap.icon className="h-4 w-4 text-primary" />
                {cap.category}
                <Badge variant="outline" className="ml-auto text-xs font-normal">
                  {cap.count}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {cap.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Company Dashboard */}
      <h2 className="mb-4 font-display text-xl font-bold">Company Dashboard</h2>
      <div className="mb-10 grid gap-4">
        {DASHBOARD_FEATURES.map((f) => (
          <div
            key={f.label}
            className="flex gap-4 rounded-xl border border-border/60 bg-card p-4"
          >
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
              <f.icon className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h3 className="font-display text-sm font-semibold">{f.label}</h3>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                {f.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* API key note */}
      <div className="rounded-xl border border-primary/20 bg-primary/5 p-5">
        <h3 className="mb-2 font-display text-sm font-semibold text-primary">
          API Key Architecture
        </h3>
        <p className="text-xs leading-relaxed text-muted-foreground">
          Keys are never stored on the server. They&apos;re saved in your browser&apos;s{" "}
          <code className="text-xs">localStorage</code> and injected per-request as HTTP
          headers or WebSocket auth messages. The backend falls back to server-side
          environment variables if no client key is provided — useful for local
          development.
        </p>
      </div>
    </div>
  );
}
