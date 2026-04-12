"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, TrendingUp, FileText, Brain, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { tickerValid } from "@/lib/utils";

const FEATURES = [
  {
    icon: FileText,
    title: "SEC Filing Analysis",
    description:
      "Deep analysis of 10-K, 10-Q, and 8-K filings. Risk factors, MD&A, balance sheets — extracted and structured by LLM.",
  },
  {
    icon: TrendingUp,
    title: "Technical Charts",
    description:
      "Interactive candlestick charts with RSI, MACD, Bollinger Bands, and pattern detection across multiple timeframes.",
  },
  {
    icon: Brain,
    title: "AI-Powered Queries",
    description:
      "Ask natural language questions about any public company. The agent decomposes complex queries into structured research steps.",
  },
];

const EXAMPLE_TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL"];

export default function HomePage() {
  const router = useRouter();
  const [ticker, setTicker] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const clean = ticker.trim().toUpperCase();
    if (!clean) {
      setError("Enter a ticker symbol");
      return;
    }
    if (!tickerValid(clean)) {
      setError("Invalid ticker format");
      return;
    }
    router.push(`/company/${clean}`);
  }

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden px-4 py-20 sm:px-6 sm:py-28 lg:py-36">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 -z-10 overflow-hidden"
        >
          <div className="absolute left-1/2 top-0 h-[600px] w-[800px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/5 blur-3xl" />
          <div className="absolute right-0 top-1/2 h-[400px] w-[400px] -translate-y-1/2 translate-x-1/2 rounded-full bg-primary/5 blur-3xl" />
        </div>

        <div className="mx-auto max-w-4xl text-center">
          <Badge
            variant="secondary"
            className="mb-6 rounded-full px-4 py-1 text-xs font-medium"
          >
            <Zap className="mr-1.5 h-3 w-3 text-primary" />
            Powered by LangGraph
          </Badge>

          <h1 className="font-display text-4xl font-bold leading-tight tracking-tight sm:text-5xl lg:text-6xl">
          Investment insights{" "}
            <span className="text-primary"></span>
            <br />
            
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-base text-muted-foreground sm:text-lg">
            Ask anything about a public company. Get structured analysis from
            SEC filings, real-time technical analysis, and web research.
          </p>

          <form
            onSubmit={handleSubmit}
            className="mx-auto mt-10 flex max-w-md flex-col gap-3 sm:flex-row"
          >
            <div className="relative flex-1">
              <Input
                value={ticker}
                onChange={(e) => {
                  setTicker(e.target.value.toUpperCase());
                  setError("");
                }}
                placeholder="Enter ticker (e.g. AAPL)"
                maxLength={10}
                className="h-12 rounded-full pl-5 pr-4 text-base font-medium uppercase tracking-widest placeholder:normal-case placeholder:tracking-normal"
                aria-label="Stock ticker"
              />
              {error && (
                <p className="absolute -bottom-5 left-4 text-xs text-destructive">
                  {error}
                </p>
              )}
            </div>
            <Button
              type="submit"
              size="lg"
              className="h-12 rounded-full px-6 font-semibold"
            >
              Analyze <ArrowRight className="ml-1.5 h-4 w-4" />
            </Button>
          </form>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
            <span className="text-xs text-muted-foreground">Try:</span>
            {EXAMPLE_TICKERS.map((t) => (
              <button
                key={t}
                onClick={() => router.push(`/company/${t}`)}
                className="rounded-full border border-border px-3 py-1 text-xs font-mono font-medium text-muted-foreground transition-colors hover:border-primary/50 hover:text-primary"
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Stats bar */}
      <section className="border-y border-border/40 bg-muted/30 px-4 py-8 sm:px-6">
        <div className="mx-auto grid max-w-4xl grid-cols-3 gap-4 text-center sm:gap-8">
          {[
            { value: "10-K / 10-Q / 8-K / 20-F", label: "SEC filings analyzed" },
            { value: "Multi-LLM", label: "Google · OpenAI · Anthropic" },
            { value: "Real-time", label: "Streaming responses" },
          ].map((s) => (
            <div key={s.label}>
              <p className="font-display text-lg font-bold text-foreground sm:text-2xl">
                {s.value}
              </p>
              <p className="mt-1 text-xs text-muted-foreground sm:text-sm">
                {s.label}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="px-4 py-20 sm:px-6">
        <div className="mx-auto max-w-5xl">
          <div className="mb-12 text-center">
            <h2 className="font-display text-2xl font-bold sm:text-3xl">
              Beyond search. Real analysis.
            </h2>
            <p className="mt-3 text-muted-foreground">
              SEC.gov gives you documents. We give you structured insight.
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-3">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="group rounded-xl border border-border/60 bg-card p-6 transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
              >
                <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <f.icon className="h-5 w-5 text-primary" />
                </div>
                <h3 className="mb-2 font-display text-base font-semibold">
                  {f.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {f.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 pb-24 sm:px-6">
        <div className="mx-auto max-w-2xl rounded-2xl border border-border/60 bg-card p-10 text-center">
          <h2 className="font-display text-2xl font-bold">
            Start analyzing in seconds
          </h2>
          <p className="mt-3 text-sm text-muted-foreground">
            Bring your own API key. No account required. All keys stored
            locally in your browser.
          </p>
          <div className="mt-6 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
            <Button
              size="lg"
              className="rounded-full px-8 font-semibold"
              onClick={() => router.push("/settings")}
            >
              Configure API Keys <ArrowRight className="ml-1.5 h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="rounded-full px-8"
              onClick={() => router.push("/about")}
            >
              Learn how it works
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
