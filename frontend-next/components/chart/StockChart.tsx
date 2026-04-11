"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";
import type { ChartResponse } from "@/types";
import type { Period, Indicator } from "@/lib/constants";

// lightweight-charts uses `window` — must be loaded client-side only
const StockChartInner = dynamic(() => import("./StockChartInner"), {
  ssr: false,
  loading: () => (
    <div className="space-y-px rounded-xl overflow-hidden border border-border/60">
      <Skeleton className="h-[340px] w-full rounded-none" />
    </div>
  ),
});

interface StockChartProps {
  data: ChartResponse;
  period: Period;
  indicators: Set<Indicator>;
}

export function StockChart({ data, period, indicators }: StockChartProps) {
  return <StockChartInner data={data} period={period} indicators={indicators} />;
}
