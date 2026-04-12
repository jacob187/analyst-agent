"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";
import { Plus, Trash2, TrendingUp, TrendingDown, Sparkles, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { useApiKeys } from "@/hooks/useApiKeys";
import { formatCurrency, formatPercent, tickerValid, cn } from "@/lib/utils";
import type { WatchlistItem, Quote } from "@/types";

marked.setOptions({ breaks: true });

interface WatchlistPanelProps {
  initialItems: WatchlistItem[];
  initialQuotes: Record<string, Quote>;
}

export function WatchlistPanel({ initialItems, initialQuotes }: WatchlistPanelProps) {
  const router = useRouter();
  const { keys } = useApiKeys();
  const [items, setItems] = useState<WatchlistItem[]>(initialItems);
  const [quotes, setQuotes] = useState<Record<string, Quote>>(initialQuotes);
  const [input, setInput] = useState("");
  const [addError, setAddError] = useState("");
  const [briefing, setBriefing] = useState<string | null>(null);
  const [briefingLoading, setBriefingLoading] = useState(false);
  const [briefingOpen, setBriefingOpen] = useState(false);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    const t = input.trim().toUpperCase();
    if (!t) return;
    if (!tickerValid(t)) { setAddError("Invalid ticker"); return; }
    if (items.some((i) => i.ticker === t)) { setAddError("Already in watchlist"); return; }
    if (items.length >= 10) { setAddError("Max 10 tickers"); return; }
    await api.addToWatchlist(t).catch(() => null);
    setInput("");
    setAddError("");
    const res = await api.watchlist().catch(() => ({ tickers: items }));
    setItems(res.tickers);
    if (res.tickers.length > 0) {
      const tickers = res.tickers.map((i) => i.ticker);
      const q = await api.quotes(tickers).catch(() => ({ quotes }));
      setQuotes(q.quotes);
    }
  }

  async function handleRemove(ticker: string) {
    await api.removeFromWatchlist(ticker).catch(() => null);
    setItems((prev) => prev.filter((i) => i.ticker !== ticker));
    setQuotes((prev) => { const n = { ...prev }; delete n[ticker]; return n; });
  }

  async function generateBriefing() {
    setBriefingLoading(true);
    setBriefingOpen(true);
    const res = await api.briefing(keys).catch(() => null);
    setBriefing(res?.briefing ?? "Failed to generate briefing.");
    setBriefingLoading(false);
  }

  return (
    <div className="space-y-4">
      {/* Add ticker */}
      <form onSubmit={handleAdd} className="flex gap-2">
        <div className="relative flex-1">
          <Input
            value={input}
            onChange={(e) => { setInput(e.target.value.toUpperCase()); setAddError(""); }}
            placeholder="Add ticker (e.g. NVDA)"
            maxLength={10}
            className="rounded-full uppercase tracking-widest placeholder:normal-case placeholder:tracking-normal"
          />
          {addError && (
            <p className="absolute -bottom-4 left-3 text-xs text-destructive">{addError}</p>
          )}
        </div>
        <Button type="submit" size="icon" className="h-9 w-9 shrink-0 rounded-full">
          <Plus className="h-4 w-4" />
        </Button>
      </form>

      {/* Briefing button */}
      {items.length > 0 && (
        <Button
          variant="outline"
          className="w-full rounded-full gap-2"
          onClick={generateBriefing}
          disabled={briefingLoading}
        >
          <Sparkles className="h-4 w-4 text-primary" />
          {briefingLoading ? "Generating briefing…" : "Generate AI Briefing"}
        </Button>
      )}

      {/* Briefing output */}
      {briefing && (
        <Card>
          <CardHeader className="pb-2">
            <button
              onClick={() => setBriefingOpen((o) => !o)}
              className="flex w-full items-center justify-between"
            >
              <CardTitle className="font-display text-sm flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                AI Market Briefing
              </CardTitle>
              <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", briefingOpen && "rotate-180")} />
            </button>
          </CardHeader>
          {briefingOpen && (
            <CardContent>
              <div
                className="prose-chat text-sm"
                // Content is sanitized with DOMPurify before injection
                // eslint-disable-next-line react/no-danger
                dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(marked.parse(briefing) as string) }}
              />
            </CardContent>
          )}
        </Card>
      )}

      {/* Ticker list */}
      {items.length === 0 ? (
        <div className="py-10 text-center">
          <p className="text-sm text-muted-foreground">Your watchlist is empty.</p>
          <p className="mt-1 text-xs text-muted-foreground">Add tickers above to track them.</p>
        </div>
      ) : (
        <div className="rounded-xl border border-border/60 bg-card overflow-hidden divide-y divide-border/40">
          {items.map((item) => {
            const q = quotes[item.ticker];
            const positive = (q?.changePercent ?? 0) >= 0;
            return (
              <div
                key={item.ticker}
                className="flex items-center justify-between px-4 py-3 hover:bg-muted/20 transition-colors"
              >
                <button
                  onClick={() => router.push(`/company/${item.ticker}`)}
                  className="flex items-center gap-3 text-left"
                >
                  <div>
                    <p className="font-display font-bold text-sm">{item.ticker}</p>
                    {item.name && (
                      <p className="text-xs text-muted-foreground truncate max-w-[150px]">{item.name}</p>
                    )}
                  </div>
                </button>
                <div className="flex items-center gap-3">
                  {q ? (
                    <div className="text-right">
                      <p className="font-medium text-sm tabular-nums">{formatCurrency(q.price)}</p>
                      <Badge
                        variant={positive ? "default" : "destructive"}
                        className="text-[10px]"
                      >
                        {positive ? <TrendingUp className="mr-0.5 h-2.5 w-2.5" /> : <TrendingDown className="mr-0.5 h-2.5 w-2.5" />}
                        {formatPercent(q.changePercent)}
                      </Badge>
                    </div>
                  ) : (
                    <Skeleton className="h-8 w-16" />
                  )}
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 rounded-lg text-muted-foreground hover:text-destructive"
                    onClick={() => handleRemove(item.ticker)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
