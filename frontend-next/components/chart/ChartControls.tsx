"use client";

import { cn } from "@/lib/utils";
import { PERIODS, INDICATORS, type Period, type Indicator } from "@/lib/constants";

interface ChartControlsProps {
  period: Period;
  indicators: Set<Indicator>;
  onPeriodChange: (p: Period) => void;
  onIndicatorToggle: (i: Indicator) => void;
}

export function ChartControls({
  period,
  indicators,
  onPeriodChange,
  onIndicatorToggle,
}: ChartControlsProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      {/* Timeframes */}
      <div className="flex items-center gap-1 rounded-lg border border-border/60 bg-muted/40 p-1">
        {PERIODS.map((p) => (
          <button
            key={p}
            onClick={() => onPeriodChange(p)}
            className={cn(
              "rounded-md px-3 py-1 text-xs font-medium transition-all",
              period === p
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {p}
          </button>
        ))}
      </div>

      {/* Indicators */}
      <div className="flex items-center gap-1">
        <span className="mr-1 text-xs text-muted-foreground">Indicators:</span>
        {INDICATORS.map((i) => (
          <button
            key={i}
            onClick={() => onIndicatorToggle(i)}
            className={cn(
              "rounded-md border px-2.5 py-1 text-xs font-medium transition-all",
              indicators.has(i)
                ? "border-primary/50 bg-primary/10 text-primary"
                : "border-border/60 text-muted-foreground hover:border-border hover:text-foreground"
            )}
          >
            {i}
          </button>
        ))}
      </div>
    </div>
  );
}
