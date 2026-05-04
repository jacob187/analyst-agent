"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TrendingUp, TrendingDown, FileText, BarChart2 } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { OverviewTab } from "./OverviewTab";
import { FilingsTab } from "./FilingsTab";
import { StockChart } from "@/components/chart/StockChart";
import { ChartControls } from "@/components/chart/ChartControls";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { api } from "@/lib/api";
import { useApiKeys } from "@/hooks/useApiKeys";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { PERIOD_MAP, type Period, type Indicator } from "@/lib/constants";
import type { CompanyProfileResponse, ChartResponse, FilingsResponse, FilingProgressEvent } from "@/types";

interface CompanyDashboardProps {
  ticker: string;
  initialSessionId?: string;
}

export function CompanyDashboard({ ticker, initialSessionId }: CompanyDashboardProps) {
  const { keys, loaded: keysLoaded, hasRequiredKeys } = useApiKeys();

  // Profile — fetched client-side so it doesn't block the initial page render
  const [profile, setProfile] = useState<CompanyProfileResponse | null>(null);

  // Chart
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [period, setPeriod] = useState<Period>("1Y");
  const [indicators, setIndicators] = useState<Set<Indicator>>(new Set(["MA", "RSI"]));

  // Filings — lazy-loaded on tab activation via SSE stream
  const [filingsData, setFilingsData] = useState<FilingsResponse | null>(null);
  const [filingsLoading, setFilingsLoading] = useState(false);
  const [filingsLoaded, setFilingsLoaded] = useState(false);
  const [filingsProgress, setFilingsProgress] = useState<FilingProgressEvent[]>([]);
  const filingsLoadedRef = useRef(false);
  filingsLoadedRef.current = filingsLoaded;
  const filingsControllerRef = useRef<AbortController | null>(null);

  const [activeTab, setActiveTab] = useState("overview");

  // Session ID — pre-fetched server-side, but may be updated by WebSocket auth
  const [sessionId] = useState<string | undefined>(initialSessionId);

  // keys ref so loadFilings always sends current keys without being a dep
  const keysRef = useRef(keys);
  keysRef.current = keys;

  useEffect(() => {
    api.profile(ticker).then(setProfile).catch(() => null);
  }, [ticker]);

  const loadFilings = useCallback(() => {
    if (filingsLoadedRef.current) return;
    setFilingsLoaded(true);
    filingsLoadedRef.current = true;
    setFilingsLoading(true);
    setFilingsProgress([]);

    filingsControllerRef.current = api.filingsStream(ticker, keysRef.current, {
      onEvent(event) {
        if (event.type === "progress") {
          setFilingsProgress((prev) => {
            const idx = prev.findIndex((p) => p.step === event.step);
            if (idx >= 0) {
              const next = [...prev];
              next[idx] = event;
              return next;
            }
            return [...prev, event];
          });
        } else if (event.type === "metadata") {
          setFilingsData({
            ticker,
            tenk: event.tenk_metadata ? { metadata: event.tenk_metadata } : undefined,
            tenq: event.tenq_metadata ? { metadata: event.tenq_metadata } : undefined,
            eightk:
              event.eightk_kind === "none"
                ? { kind: "none" }
                : { kind: event.eightk_kind, metadata: event.eightk_metadata ?? undefined },
          });
        } else if (event.type === "section") {
          setFilingsData((prev) => {
            if (!prev) return prev;
            if (event.form === "10-K") {
              return { ...prev, tenk: { ...prev.tenk!, [event.key]: event.data } };
            }
            if (event.form === "10-Q") {
              return { ...prev, tenq: { ...prev.tenq!, [event.key]: event.data } };
            }
            if (event.form === "8-K") {
              // Both "earnings" and "event" keys go into the same eightk slot —
              // the kind discriminator was set when the metadata event arrived.
              return { ...prev, eightk: { ...prev.eightk!, analysis: event.data } };
            }
            return prev;
          });
        } else if (event.type === "complete" || event.type === "error") {
          setFilingsLoading(false);
        }
      },
      onError() {
        setFilingsLoading(false);
      },
    });
  }, [ticker]);

  // Abort in-flight SSE stream on unmount
  useEffect(() => {
    return () => {
      filingsControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (activeTab === "filings" && keysLoaded && !filingsLoadedRef.current) {
      loadFilings();
    }
  }, [activeTab, keysLoaded, loadFilings]);

  const loadChart = useCallback(
    (p: Period) => {
      setChartLoading(true);
      api
        .chart(ticker, PERIOD_MAP[p])
        .then(setChartData)
        .catch(() => null)
        .finally(() => setChartLoading(false));
    },
    [ticker]
  );

  function handlePeriodChange(p: Period) {
    setPeriod(p);
    loadChart(p);
  }

  function handleIndicatorToggle(i: Indicator) {
    setIndicators((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }

  const quote = profile?.quote;
  const changePositive = (quote?.change ?? 0) >= 0;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
            <BarChart2 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-display text-2xl font-bold">{ticker}</h1>
              {profile?.company?.name && (
                <span className="text-sm text-muted-foreground">
                  {profile.company.name}
                </span>
              )}
            </div>
            {quote && (
              <div className="flex items-center gap-2">
                <span className="font-display text-lg font-semibold">
                  {formatCurrency(quote.price)}
                </span>
                <Badge
                  variant={changePositive ? "default" : "destructive"}
                  className="text-xs"
                >
                  {changePositive ? (
                    <TrendingUp className="mr-1 h-3 w-3" />
                  ) : (
                    <TrendingDown className="mr-1 h-3 w-3" />
                  )}
                  {formatPercent(quote.changePercent)}
                </Badge>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="rounded-xl">
          <TabsTrigger value="overview" className="rounded-lg text-sm">
            Overview
          </TabsTrigger>
          <TabsTrigger value="filings" className="rounded-lg text-sm">
            <FileText className="mr-1.5 h-3.5 w-3.5" />
            Filings
          </TabsTrigger>
          <TabsTrigger
            value="chart"
            onClick={() => !chartData && loadChart(period)}
            className="rounded-lg text-sm"
          >
            <BarChart2 className="mr-1.5 h-3.5 w-3.5" />
            Chart + Chat
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-4">
          <OverviewTab data={profile} loading={profile === null} />
        </TabsContent>

        <TabsContent value="filings" className="mt-4">
          <FilingsTab data={filingsData} loading={filingsLoading} progressSteps={filingsProgress} />
        </TabsContent>

        <TabsContent value="chart" className="mt-4">
          <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
            <div className="space-y-3">
              <ChartControls
                period={period}
                indicators={indicators}
                onPeriodChange={handlePeriodChange}
                onIndicatorToggle={handleIndicatorToggle}
              />
              {chartLoading || !chartData ? (
                <Skeleton className="h-[400px] rounded-xl" />
              ) : (
                <StockChart data={chartData} period={period} indicators={indicators} />
              )}
            </div>
            <div className="h-[560px]">
              {keysLoaded && !hasRequiredKeys() ? (
                <div className="flex h-full flex-col items-center justify-center gap-4 rounded-xl border border-border/60 bg-card p-8 text-center">
                  <p className="text-sm font-medium">API keys required</p>
                  <p className="text-xs text-muted-foreground">
                    Configure a provider key and SEC EDGAR User-Agent to use the AI analyst.
                  </p>
                  <a
                    href="/settings"
                    className="rounded-lg bg-primary px-4 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90"
                  >
                    Go to Settings
                  </a>
                </div>
              ) : (
                <ChatWindow ticker={ticker} keys={keys} initialSessionId={sessionId} />
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
