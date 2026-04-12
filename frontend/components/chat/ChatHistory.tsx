"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, Eye, ArrowRight, Trash2, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatDate, cn } from "@/lib/utils";
import type { Session, TickerSummary } from "@/types";

interface ChatHistoryProps {
  initialTickers: TickerSummary[];
}

export function ChatHistory({ initialTickers }: ChatHistoryProps) {
  const router = useRouter();
  const [tickers, setTickers] = useState<TickerSummary[]>(initialTickers);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Record<string, Session[]>>({});
  const [loadingSessions, setLoadingSessions] = useState<string | null>(null);

  async function toggle(ticker: string) {
    if (expanded === ticker) {
      setExpanded(null);
      return;
    }
    setExpanded(ticker);
    if (!sessions[ticker]) {
      setLoadingSessions(ticker);
      const res = await api.sessions(ticker).catch(() => ({ sessions: [] }));
      setSessions((prev) => ({ ...prev, [ticker]: res.sessions }));
      setLoadingSessions(null);
    }
  }

  async function deleteSession(sessionId: string, ticker: string) {
    await api.deleteSession(sessionId, ticker).catch(() => null);
    setSessions((prev) => ({
      ...prev,
      [ticker]: prev[ticker]?.filter((s) => s.id !== sessionId) ?? [],
    }));
    setTickers((prev) =>
      prev
        .map((t) =>
          t.ticker === ticker ? { ...t, session_count: t.session_count - 1 } : t
        )
        .filter((t) => t.session_count > 0)
    );
  }

  if (!tickers.length) {
    return (
      <div className="py-16 text-center">
        <MessageSquare className="mx-auto mb-3 h-10 w-10 text-muted-foreground/40" />
        <p className="text-sm text-muted-foreground">No sessions yet.</p>
        <Button
          variant="outline"
          size="sm"
          className="mt-4 rounded-full"
          onClick={() => router.push("/")}
        >
          Start analyzing
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {tickers.map((t) => (
        <div
          key={t.ticker}
          className="rounded-xl border border-border/60 bg-card overflow-hidden"
        >
          <button
            onClick={() => toggle(t.ticker)}
            className="flex w-full items-center justify-between px-4 py-3.5 hover:bg-muted/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className="font-display font-bold">{t.ticker}</span>
              <Badge variant="secondary" className="text-xs">
                {t.session_count} session{t.session_count !== 1 ? "s" : ""}
              </Badge>
              <span className="hidden text-xs text-muted-foreground sm:block">
                {formatDate(t.last_active)}
              </span>
            </div>
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                expanded === t.ticker && "rotate-180"
              )}
            />
          </button>

          {expanded === t.ticker && (
            <div className="border-t border-border/40">
              {loadingSessions === t.ticker ? (
                <div className="p-3 space-y-2">
                  {[...Array(2)].map((_, i) => (
                    <Skeleton key={i} className="h-10 rounded-lg" />
                  ))}
                </div>
              ) : (
                <div className="divide-y divide-border/40">
                  {sessions[t.ticker]?.map((s) => (
                    <div
                      key={s.id}
                      className="flex items-center justify-between px-4 py-2.5"
                    >
                      <div className="text-xs text-muted-foreground">
                        <span>{formatDate(s.created_at)}</span>
                        {s.model && (
                          <span className="ml-2 text-[10px]">· {s.model}</span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 gap-1 rounded-lg px-2 text-xs"
                          onClick={() =>
                            router.push(`/session/${s.id}?ticker=${t.ticker}`)
                          }
                        >
                          <Eye className="h-3 w-3" />
                          View
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 gap-1 rounded-lg px-2 text-xs text-primary hover:text-primary"
                          onClick={() =>
                            router.push(`/company/${t.ticker}?session=${s.id}`)
                          }
                        >
                          <ArrowRight className="h-3 w-3" />
                          Continue
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 rounded-lg p-0 text-muted-foreground hover:text-destructive"
                          onClick={() => deleteSession(s.id, t.ticker)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
