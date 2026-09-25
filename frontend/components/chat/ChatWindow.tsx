"use client";

import { useEffect, useRef, useState } from "react";
import { Send, Wifi, WifiOff, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "./ChatMessage";
import { useWebSocket } from "@/hooks/useWebSocket";
import { cn } from "@/lib/utils";
import type { ApiKeys } from "@/types";

// Each chains several tool families (SEC filings, stock/technicals, market
// macro) so the planner decomposes it into steps. No Tavily tools — trial
// visitors don't get web research.
const DEMO_PROMPTS = [
  {
    label: "Risks vs. balance sheet",
    query:
      "What are the biggest risk factors in the latest 10-K, how exposed is the balance sheet to them, and how is the stock's technical trend pricing that risk?",
  },
  {
    label: "8-K vs. outlook",
    query:
      "What did the latest 8-K report, how does it line up with management's outlook in the MD&A, and how has the stock reacted across timeframes since?",
  },
  {
    label: "Macro impact",
    query:
      "What is the current macro backdrop (rates, VIX, major indices), how does it affect this company's business model, and what does that mean for its valuation metrics?",
  },
  {
    label: "Legal & cyber exposure",
    query:
      "What legal proceedings and cybersecurity risks are disclosed, how material are they relative to the company's cash and debt, and has the chart shown any bearish patterns as a result?",
  },
  {
    label: "Investment case",
    query:
      "What drives this business, how are margins and cash flow trending according to the MD&A, how does the market backdrop affect that, and what does the technical setup imply for entry timing?",
  },
];

interface ChatWindowProps {
  ticker: string;
  keys: ApiKeys;
  initialSessionId?: string;
}

export function ChatWindow({ ticker, keys, initialSessionId }: ChatWindowProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  // History loading is handled inside the hook — no callback needed here
  const { messages, status, sendMessage, freeTrial } = useWebSocket({
    ticker,
    keys,
    sessionId: initialSessionId,
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    const text = input.trim();
    if (!text || status !== "connected") return;
    sendMessage(text);
    setInput("");
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const statusColor = {
    connecting: "text-yellow-500",
    connected: "text-primary",
    disconnected: "text-muted-foreground",
    error: "text-destructive",
  }[status];

  const StatusIcon =
    status === "connecting" ? Loader2 : status === "connected" ? Wifi : WifiOff;

  return (
    <div className="flex h-full flex-col rounded-xl border border-border/60 bg-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="font-display text-sm font-semibold">{ticker}</span>
          <span className="text-xs text-muted-foreground">AI Analyst</span>
        </div>
        <div className={cn("flex items-center gap-1.5 text-xs", statusColor)}>
          <StatusIcon
            className={cn("h-3.5 w-3.5", status === "connecting" && "animate-spin")}
          />
          <span className="capitalize">{status}</span>
        </div>
      </div>

      {freeTrial && (
        <div className="border-b border-border/60 bg-primary/5 px-4 py-1.5 text-center text-[11px] text-muted-foreground">
          Free trial — up to {freeTrial.queries} queries per day.{" "}
          <a href="/settings" className="underline hover:text-foreground">
            Sign in or add your own key
          </a>{" "}
          for unlimited access.
        </div>
      )}

      {/* Messages */}
      <ScrollArea className="flex-1 p-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center py-12 text-center">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
              <Send className="h-5 w-5 text-primary" />
            </div>
            <p className="text-sm font-medium">Ask anything about {ticker}</p>
            <p className="mt-1 text-xs text-muted-foreground">Or try one of these:</p>
            <div className="mt-3 flex max-w-md flex-wrap justify-center gap-2">
              {DEMO_PROMPTS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  title={p.query}
                  disabled={status !== "connected"}
                  onClick={() => sendMessage(p.query)}
                  className="rounded-full border border-border/60 bg-background px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border/60 p-3">
        <div className="flex items-end gap-2 rounded-xl border border-border/60 bg-background px-3 py-2 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              status === "connected"
                ? "Ask about this company…"
                : status === "connecting"
                ? "Connecting…"
                : "Disconnected"
            }
            disabled={status !== "connected"}
            rows={1}
            className="flex-1 resize-none overflow-y-auto bg-transparent text-sm leading-5 outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
            style={{ maxHeight: "200px" }}
          />
          <Button
            size="icon"
            className="h-7 w-7 shrink-0 rounded-lg"
            disabled={!input.trim() || status !== "connected"}
            onClick={handleSend}
          >
            <Send className="h-3.5 w-3.5" />
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
