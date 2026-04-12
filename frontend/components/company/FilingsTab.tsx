"use client";

import { Check, FileX2, Loader2, XCircle, Zap } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { FilingSection } from "./FilingSection";
import type { FilingsResponse, FilingProgressEvent } from "@/types";

// ─── Step label mapping ────────────────────────────────────────────────────────

const STEP_LABELS: Record<string, string> = {
  edgar_fetch: "Fetching filings from EDGAR",
  "10-K/risk_10k": "10-K · Risk Factors",
  "10-K/mda_10k": "10-K · Management Discussion & Analysis",
  "10-K/balance": "10-K · Balance Sheet",
  "10-K/income_stmt": "10-K · Income Statement",
  "10-K/cashflow": "10-K · Cash Flow",
  "10-K/business": "10-K · Business Overview",
  "10-K/cybersecurity": "10-K · Cybersecurity",
  "10-K/legal": "10-K · Legal Proceedings",
  "10-K/market_risk": "10-K · Market Risk",
  "10-Q/risk_10q": "10-Q · Risk Factors",
  "10-Q/mda_10q": "10-Q · MD&A",
  "8-K/earnings": "8-K · Earnings Analysis",
};

// ─── Progress activity log ─────────────────────────────────────────────────────

function ProgressRow({ step }: { step: FilingProgressEvent }) {
  const label = STEP_LABELS[step.step] ?? step.step;
  const active = step.status === "fetching" || step.status === "processing";
  const cached = step.status === "cached";
  const failed = step.status === "failed";

  return (
    <div className="flex items-center gap-2 text-xs">
      {active ? (
        <Loader2 className="h-3 w-3 shrink-0 animate-spin text-muted-foreground" />
      ) : failed ? (
        <XCircle className="h-3 w-3 shrink-0 text-destructive" />
      ) : cached ? (
        <Zap className="h-3 w-3 shrink-0 text-blue-400" />
      ) : (
        <Check className="h-3 w-3 shrink-0 text-emerald-500" />
      )}

      <span className={active ? "text-foreground" : "text-muted-foreground"}>
        {label}
      </span>

      {cached && (
        <span className="rounded bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-400">
          cached
        </span>
      )}

      {step.duration !== undefined && (
        <span className="ml-auto tabular-nums text-muted-foreground">
          {step.duration}s
        </span>
      )}
    </div>
  );
}

// ─── Skeleton for a section that's known-pending ───────────────────────────────

function PendingSection({ title }: { title: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="font-display text-sm font-semibold text-muted-foreground">
            {title}
          </span>
          <Skeleton className="h-4 w-16 rounded-full" />
        </div>
        <Skeleton className="h-4 w-4 rounded" />
      </div>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

interface FilingsTabProps {
  data: FilingsResponse | null;
  loading: boolean;
  progressSteps: FilingProgressEvent[];
}

export function FilingsTab({ data, loading, progressSteps }: FilingsTabProps) {
  // Before metadata arrives, show generic skeletons
  if (loading && !data) {
    return (
      <div className="space-y-3">
        {progressSteps.length > 0 ? (
          <div className="rounded-xl border bg-muted/30 p-3 space-y-2">
            {progressSteps.map((s) => (
              <ProgressRow key={s.step} step={s} />
            ))}
          </div>
        ) : (
          [...Array(3)].map((_, i) => (
            <Skeleton key={i} className="h-16 rounded-xl" />
          ))
        )}
      </div>
    );
  }

  if (!data) return null;

  const hasFilings = data.tenk || data.tenq || data.earnings?.has_earnings;

  if (!loading && !hasFilings) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <FileX2 className="h-10 w-10 text-muted-foreground/50" />
        <p className="mt-3 text-sm font-medium text-muted-foreground">
          No SEC filings found
        </p>
        <p className="mt-1 text-xs text-muted-foreground/70">
          This ticker may be an ETF, mutual fund, or foreign issuer without standard SEC filings.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Activity log — visible while loading, dismissed on complete */}
      {loading && progressSteps.length > 0 && (
        <div className="rounded-xl border bg-muted/30 p-3 space-y-2">
          {progressSteps.map((s) => (
            <ProgressRow key={s.step} step={s} />
          ))}
        </div>
      )}

      {/* Annual report — 10-K for domestic filers, 20-F for foreign filers */}
      {data.tenk && (
        <>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Annual Report ({data.tenk.metadata?.form_type ?? "10-K"})
          </p>
          {data.tenk.risk_10k ? (
            <FilingSection
              title="Risk Factors"
              analysis={data.tenk.risk_10k}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Risk Factors" />
          ) : null}

          {data.tenk.mda_10k ? (
            <FilingSection
              title="Management Discussion & Analysis"
              analysis={data.tenk.mda_10k}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Management Discussion & Analysis" />
          ) : null}

          {data.tenk.balance ? (
            <FilingSection
              title="Balance Sheet Analysis"
              analysis={data.tenk.balance}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Balance Sheet Analysis" />
          ) : null}

          {data.tenk.income_stmt ? (
            <FilingSection
              title="Income Statement"
              analysis={data.tenk.income_stmt}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Income Statement" />
          ) : null}

          {data.tenk.cashflow ? (
            <FilingSection
              title="Cash Flow Statement"
              analysis={data.tenk.cashflow}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Cash Flow Statement" />
          ) : null}

          {data.tenk.business ? (
            <FilingSection
              title="Business Overview"
              analysis={data.tenk.business}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Business Overview" />
          ) : null}

          {data.tenk.market_risk ? (
            <FilingSection
              title="Market Risk"
              analysis={data.tenk.market_risk}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Market Risk" />
          ) : null}

          {data.tenk.legal ? (
            <FilingSection
              title="Legal Proceedings"
              analysis={data.tenk.legal}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Legal Proceedings" />
          ) : null}

          {data.tenk.cybersecurity ? (
            <FilingSection
              title="Cybersecurity"
              analysis={data.tenk.cybersecurity}
              metadata={data.tenk.metadata}
            />
          ) : loading ? (
            <PendingSection title="Cybersecurity" />
          ) : null}
        </>
      )}

      {/* 10-Q sections */}
      {data.tenq && (
        <>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Quarterly Report (10-Q)
          </p>
          {data.tenq.risk_10q ? (
            <FilingSection
              title="Risk Factors"
              analysis={data.tenq.risk_10q}
              metadata={data.tenq.metadata}
            />
          ) : loading ? (
            <PendingSection title="Risk Factors" />
          ) : null}

          {data.tenq.mda_10q ? (
            <FilingSection
              title="MD&A"
              analysis={data.tenq.mda_10q}
              metadata={data.tenq.metadata}
            />
          ) : loading ? (
            <PendingSection title="MD&A" />
          ) : null}
        </>
      )}

      {/* 8-K earnings */}
      {data.earnings?.has_earnings && (
        <>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Earnings (8-K)
          </p>
          {data.earnings.analysis ? (
            <FilingSection
              title="Earnings Analysis"
              analysis={data.earnings.analysis}
              metadata={data.earnings.metadata}
            />
          ) : loading ? (
            <PendingSection title="Earnings Analysis" />
          ) : null}
        </>
      )}
    </div>
  );
}
