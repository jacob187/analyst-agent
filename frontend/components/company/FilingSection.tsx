"use client";

import { useState } from "react";
import { ExternalLink, ChevronDown, TrendingUp, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { FilingAnalysis, FilingMetadata } from "@/types";

function SentimentBadge({ score }: { score?: number }) {
  if (score == null) return null;
  const label = score > 0.3 ? "Positive" : score < -0.3 ? "Negative" : "Neutral";
  const variant =
    label === "Positive" ? "default" : label === "Negative" ? "destructive" : "secondary";
  return (
    <Badge variant={variant} className="text-xs">
      {label} {score > 0 ? "+" : ""}
      {score.toFixed(2)}
    </Badge>
  );
}

interface FilingSectionProps {
  title: string;
  analysis: FilingAnalysis | undefined;
  metadata?: FilingMetadata;
}

export function FilingSection({ title, analysis, metadata }: FilingSectionProps) {
  const [open, setOpen] = useState(false);
  if (!analysis) return null;

  const sentiment = analysis.sentiment_score as number | undefined;
  const summary = analysis.summary as string | undefined;
  const risks = analysis.risks as string[] | undefined;
  const keyMetrics = analysis.key_metrics as string[] | undefined;
  const outlook = analysis.outlook as string | undefined;
  const redFlags = analysis.red_flags as string[] | undefined;
  // MaterialEventAnalysis fields (8-K event)
  const keyPoints = analysis.key_points as string[] | undefined;
  const impactAssessment = analysis.impact_assessment as string | undefined;

  return (
    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="font-display text-sm font-semibold">{title}</span>
          {sentiment != null && <SentimentBadge score={sentiment} />}
          {metadata?.filing_date && (
            <span className="text-xs text-muted-foreground">{metadata.filing_date}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {metadata?.edgar_url && (
            <a
              href={metadata.edgar_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-muted-foreground hover:text-primary transition-colors"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              open && "rotate-180"
            )}
          />
        </div>
      </button>

      {open && (
        <div className="border-t border-border/40 px-4 pb-4 pt-3 space-y-3">
          {summary && (
            <p className="text-sm leading-relaxed text-muted-foreground">{summary}</p>
          )}

          {risks && risks.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Risk Factors
              </p>
              <ul className="space-y-1.5">
                {risks.slice(0, 5).map((r, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-yellow-500" />
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {keyMetrics && keyMetrics.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Key Metrics
              </p>
              <ul className="space-y-1">
                {keyMetrics.map((m, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-foreground/80">
                    <span className="mt-0.5 shrink-0 text-muted-foreground">·</span>
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {keyPoints && keyPoints.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Key Points
              </p>
              <ul className="space-y-1">
                {keyPoints.map((p, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-foreground/80">
                    <span className="mt-0.5 shrink-0 text-muted-foreground">·</span>
                    <span>{p}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {redFlags && redFlags.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-destructive">
                Red Flags
              </p>
              <ul className="space-y-1">
                {redFlags.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-destructive">
                    <span>·</span>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {outlook && (
            <div className="flex items-start gap-2 rounded-lg bg-primary/5 border border-primary/20 px-3 py-2">
              <TrendingUp className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
              <p className="text-xs leading-relaxed">{outlook}</p>
            </div>
          )}

          {impactAssessment && (
            <div className="flex items-start gap-2 rounded-lg bg-primary/5 border border-primary/20 px-3 py-2">
              <TrendingUp className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
              <p className="text-xs leading-relaxed">{impactAssessment}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
