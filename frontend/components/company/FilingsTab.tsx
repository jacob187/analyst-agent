"use client";

import { Check, FileX2, Loader2, Lock, XCircle, Zap } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { FilingSection } from "./FilingSection";
import type {
  FilingsResponse,
  FilingProgressEvent,
  FilingAnalysis,
  FilingMetadata,
} from "@/types";

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
  "8-K/event": "8-K · Material Event",
};

// ─── Progress activity log ─────────────────────────────────────────────────────

function ProgressRow({ step }: { step: FilingProgressEvent }) {
  const label = STEP_LABELS[step.step] ?? step.step;
  const active = step.status === "fetching" || step.status === "processing";
  const cached = step.status === "cached";
  const failed = step.status === "failed";
  const needsKey = step.status === "needs_key";

  return (
    <div className="flex items-center gap-2 text-xs">
      {active ? (
        <Loader2 className="h-3 w-3 shrink-0 animate-spin text-muted-foreground" />
      ) : failed ? (
        <XCircle className="h-3 w-3 shrink-0 text-destructive" />
      ) : needsKey ? (
        <Lock className="h-3 w-3 shrink-0 text-amber-400" />
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

      {needsKey && (
        <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400">
          sign in to generate
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

// ─── Locked section — cached miss with no key (anon / keyless) ─────────────────

function LockedSection({ title }: { title: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3">
        <Lock className="h-3.5 w-3.5 shrink-0 text-amber-400" />
        <span className="font-display text-sm font-semibold text-muted-foreground">
          {title}
        </span>
        <span className="ml-auto text-xs text-muted-foreground">
          Sign in or add a key to generate
        </span>
      </div>
    </div>
  );
}

// Banner shown when some sections couldn't be generated for lack of a key.
function LockedAnalysisBanner() {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3">
      <Lock className="h-4 w-4 shrink-0 text-amber-400" />
      <p className="text-xs text-muted-foreground">
        Some AI analyses aren&apos;t cached yet. <strong className="text-foreground">Sign in</strong>{" "}
        to generate them with our model, or add your own API key.
      </p>
      <a
        href="/settings"
        className="ml-auto rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
      >
        Settings
      </a>
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

  const hasFilings =
    data.tenk || data.tenq || (data.eightk && data.eightk.kind !== "none");

  // A keyless caller (anon, or signed-in without a key) gets cached sections;
  // uncached ones come back as `needs_key`. Track WHICH steps need a key (keyed
  // `${form}/${type}`, e.g. "10-K/risk_10k") so only those slots lock — a
  // section the filing simply doesn't contain emits no step and stays absent.
  const needsKeySteps = new Set(
    progressSteps.filter((s) => s.status === "needs_key").map((s) => s.step),
  );
  const anyNeedsKey = needsKeySteps.size > 0;

  // One section slot: real analysis → pending skeleton → locked prompt → nothing.
  const sect = (
    title: string,
    step: string,
    analysis: FilingAnalysis | undefined,
    metadata: FilingMetadata | undefined,
  ) => {
    if (analysis) return <FilingSection title={title} analysis={analysis} metadata={metadata} />;
    if (loading) return <PendingSection title={title} />;
    if (needsKeySteps.has(step)) return <LockedSection title={title} />;
    return null;
  };

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

      {/* Persistent CTA once streaming is done and some sections were gated */}
      {!loading && anyNeedsKey && <LockedAnalysisBanner />}

      {/* Annual report — 10-K for domestic filers, 20-F for foreign filers */}
      {data.tenk && (
        <>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Annual Report ({data.tenk.metadata?.form_type ?? "10-K"})
          </p>
          {sect("Risk Factors", "10-K/risk_10k", data.tenk.risk_10k, data.tenk.metadata)}
          {sect("Management Discussion & Analysis", "10-K/mda_10k", data.tenk.mda_10k, data.tenk.metadata)}
          {sect("Balance Sheet Analysis", "10-K/balance", data.tenk.balance, data.tenk.metadata)}
          {sect("Income Statement", "10-K/income_stmt", data.tenk.income_stmt, data.tenk.metadata)}
          {sect("Cash Flow Statement", "10-K/cashflow", data.tenk.cashflow, data.tenk.metadata)}
          {sect("Business Overview", "10-K/business", data.tenk.business, data.tenk.metadata)}
          {sect("Market Risk", "10-K/market_risk", data.tenk.market_risk, data.tenk.metadata)}
          {sect("Legal Proceedings", "10-K/legal", data.tenk.legal, data.tenk.metadata)}
          {sect("Cybersecurity", "10-K/cybersecurity", data.tenk.cybersecurity, data.tenk.metadata)}
        </>
      )}

      {/* 10-Q sections */}
      {data.tenq && (
        <>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Quarterly Report (10-Q)
          </p>
          {sect("Risk Factors", "10-Q/risk_10q", data.tenq.risk_10q, data.tenq.metadata)}
          {sect("MD&A", "10-Q/mda_10q", data.tenq.mda_10q, data.tenq.metadata)}
        </>
      )}

      {/* 8-K — earnings or material event (leadership change, M&A, cyber, Reg FD, etc.) */}
      {data.eightk && data.eightk.kind === "earnings" && (
        <>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Earnings (8-K)
          </p>
          {sect("Earnings Analysis", "8-K/earnings", data.eightk.analysis, data.eightk.metadata)}
        </>
      )}

      {data.eightk && data.eightk.kind === "event" && (
        <>
          <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Material Event (8-K)
          </p>
          {sect(
            typeof data.eightk.analysis?.event_type === "string" && data.eightk.analysis.event_type
              ? `Material Event — ${data.eightk.analysis.event_type}`
              : "Material Event",
            "8-K/event",
            data.eightk.analysis,
            data.eightk.metadata,
          )}
        </>
      )}
    </div>
  );
}
