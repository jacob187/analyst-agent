import { BarChart2, Brain, Database, Zap, GitBranch, Server } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const STACK = [
  {
    icon: Brain,
    name: "LangGraph",
    role: "Agent orchestration",
    desc: "StateGraph with router → simple ReAct or complex planner → step executor → synthesizer.",
  },
  {
    icon: Server,
    name: "FastAPI",
    role: "Backend API",
    desc: "REST + WebSocket endpoints. Async, streaming responses, SQLite persistence.",
  },
  {
    icon: Database,
    name: "SEC EDGAR (edgartools)",
    role: "Filing data",
    desc: "Fetches 10-K, 10-Q, 8-K filings. LLM analysis cached per accession number.",
  },
  {
    icon: BarChart2,
    name: "yfinance + TradingView",
    role: "Market data + charts",
    desc: "Price history, technicals, pattern detection. Rendered with Lightweight Charts.",
  },
  {
    icon: GitBranch,
    name: "Multi-provider LLM",
    role: "AI backbone",
    desc: "Supports Google Gemini, OpenAI, and Anthropic via LangChain. Thinking mode supported.",
  },
  {
    icon: Zap,
    name: "Next.js 16 + shadcn/ui",
    role: "Frontend",
    desc: "App Router, Turbopack, Tailwind CSS, next-themes. Dark and light mode.",
  },
];

const GRAPH = `
START
  └─▶ [ROUTER]
       ├─▶ "simple"  ──▶ [REACT AGENT] ──▶ END
       └─▶ "complex" ──▶ [PLANNER]
                               └─▶ [STEP EXECUTOR] ⟵──┐
                                         └─▶ (loop)  ──┘
                                         └─▶ [SYNTHESIZER] ──▶ END
`.trim();

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

      {/* Graph architecture */}
      <div className="mb-10 rounded-xl border border-border/60 bg-card p-5">
        <h2 className="mb-3 font-display text-base font-semibold">
          Graph Architecture
        </h2>
        <pre className="overflow-x-auto rounded-lg bg-muted px-4 py-3 text-xs leading-relaxed text-muted-foreground font-mono">
          {GRAPH}
        </pre>
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          Complex queries are automatically decomposed into steps by the planner
          node. Each step calls relevant tools (SEC filings, price data, web
          research) and the synthesizer assembles the final response.
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

      {/* Tech stack */}
      <h2 className="mb-4 font-display text-xl font-bold">Tech Stack</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        {STACK.map((s) => (
          <Card key={s.name} className="border-border/60">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 font-display text-sm">
                <s.icon className="h-4 w-4 text-primary" />
                {s.name}
                <Badge variant="outline" className="ml-auto text-xs font-normal">
                  {s.role}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs leading-relaxed text-muted-foreground">{s.desc}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* API key note */}
      <div className="mt-8 rounded-xl border border-primary/20 bg-primary/5 p-5">
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
