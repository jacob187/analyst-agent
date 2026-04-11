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
import type { CompanyProfileResponse, ChartResponse, FilingsResponse } from "@/types";

interface CompanyDashboardProps {
  ticker: string;
  initialSessionId?: string;
}

export function CompanyDashboard({ ticker, initialSessionId }: CompanyDashboardProps) {
  const { keys, loaded: keysLoaded } = useApiKeys();

  // Profile
  const [profile, setProfile] = useState<CompanyProfileResponse | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);

  // Chart
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [period, setPeriod] = useState<Period>("1Y");
  const [indicators, setIndicators] = useState<Set<Indicator>>(new Set(["MA", "RSI"]));

  // Filings — track whether a fetch is in-flight or done
  const [filingsData, setFilingsData] = useState<FilingsResponse | null>(null);
  const [filingsLoading, setFilingsLoading] = useState(false);
  const [filingsLoaded, setFilingsLoaded] = useState(false);
  // ref so the effect closure always sees the current value without re-triggering
  const filingsLoadedRef = useRef(false);

  // Active tab
  const [activeTab, setActiveTab] = useState("overview");

  // Session
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId);

  // Keep ref in sync
  filingsLoadedRef.current = filingsLoaded;

  // keys ref so loadFilings always sends current keys without being a dep
  const keysRef = useRef(keys);
  keysRef.current = keys;

  useEffect(() => {
    setProfileLoading(true);
    api.profile(ticker).then(setProfile).catch(() => null).finally(() => setProfileLoading(false));
  }, [ticker]);

  const loadFilings = useCallback(() => {
    if (filingsLoadedRef.current) return; // already fetching or done
    setFilingsLoaded(true);
    filingsLoadedRef.current = true;
    setFilingsLoading(true);
    api
      .filings(ticker, keysRef.current)
      .then(setFilingsData)
      .catch(() => null)
      .finally(() => setFilingsLoading(false));
  }, [ticker]);

  // Auto-fetch filings when the Filings tab becomes active and keys are ready.
  // This handles both the first visit and returning after navigation (remount).
  useEffect(() => {
    if (activeTab === "filings" && keysLoaded && !filingsLoadedRef.current) {
      loadFilings();
    }
  }, [activeTab, keysLoaded, loadFilings]);

  const loadChart = useCallback((p: Period) => {
    setChartLoading(true);
    api.chart(ticker, PERIOD_MAP[p]).then(setChartData).catch(() => null).finally(() => setChartLoading(false));
  }, [ticker]);

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

  useEffect(() => {
    if (initialSessionId) return;
    api.sessionByTicker(ticker).then((res) => {
      if (res.session?.id) setSessionId(res.session.id);
    }).catch(() => null);
  }, [ticker, initialSessionId]);

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
              {profileLoading ? (
                <Skeleton className="h-5 w-32" />
              ) : (
                profile?.company?.name && (
                  <span className="text-sm text-muted-foreground">{profile.company.name}</span>
                )
              )}
            </div>
            {quote && (
              <div className="flex items-center gap-2">
                <span className="font-display text-lg font-semibold">{formatCurrency(quote.price)}</span>
                <Badge variant={changePositive ? "default" : "destructive"} className="text-xs">
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
          <OverviewTab data={profile} loading={profileLoading} />
        </TabsContent>

        <TabsContent value="filings" className="mt-4">
          <FilingsTab data={filingsData} loading={filingsLoading} />
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
              <ChatWindow ticker={ticker} keys={keys} initialSessionId={sessionId} />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
