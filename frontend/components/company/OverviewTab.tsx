import { Globe, Users, TrendingUp, TrendingDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency, formatPercent } from "@/lib/utils";
import type { CompanyProfileResponse, Pattern } from "@/types";

interface OverviewTabProps {
  data: CompanyProfileResponse | null;
  loading: boolean;
}

function MetricRow({
  label,
  value,
}: {
  label: string;
  value: string | number | null | undefined;
}) {
  const display =
    value == null || value === "" ? "—" : String(value);
  return (
    <div className="flex items-center justify-between py-2 text-sm border-b border-border/40 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{display}</span>
    </div>
  );
}

function PatternBadge({ pattern }: { pattern: Pattern }) {
  const isBull = pattern.direction === "bullish";
  return (
    <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-2 text-xs">
      <div className="flex items-center gap-2">
        {isBull ? (
          <TrendingUp className="h-3.5 w-3.5 text-primary" />
        ) : (
          <TrendingDown className="h-3.5 w-3.5 text-destructive" />
        )}
        <span className="font-medium">{pattern.type}</span>
        {pattern.status && (
          <span className="text-muted-foreground">· {pattern.status}</span>
        )}
      </div>
      <Badge
        variant={isBull ? "default" : "destructive"}
        className="text-[10px]"
      >
        {Math.round((pattern.confidence ?? 0) * 100)}%
      </Badge>
    </div>
  );
}

export function OverviewTab({ data, loading }: OverviewTabProps) {
  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[...Array(6)].map((_, i) => (
          <Skeleton key={i} className="h-48 rounded-xl" />
        ))}
      </div>
    );
  }

  if (!data) return null;

  const { company, metrics, technicals, patterns, regime } = data;
  const rsi = (technicals?.rsi as { value?: number } | undefined)?.value;
  const macd = technicals?.macd as
    | { value?: number; signal?: number; histogram?: number }
    | undefined;
  const adx = (technicals?.adx as { value?: number } | undefined)?.value;
  const bb = technicals?.bollinger_bands as
    | { position?: string }
    | undefined;

  return (
    <div className="space-y-4">
      {/* Company header */}
      <div className="rounded-xl border border-border/60 bg-card p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-xl font-bold">{company.name}</h2>
            <div className="mt-1 flex flex-wrap gap-2">
              {company.sector && (
                <Badge variant="secondary" className="text-xs">
                  {company.sector}
                </Badge>
              )}
              {company.industry && (
                <Badge variant="outline" className="text-xs">
                  {company.industry}
                </Badge>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {company.employees > 0 && (
              <span className="flex items-center gap-1">
                <Users className="h-3.5 w-3.5" />
                {company.employees.toLocaleString()} employees
              </span>
            )}
            {company.website && /^https?:\/\//i.test(company.website) && (
              <a
                href={company.website}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-primary transition-colors"
              >
                <Globe className="h-3.5 w-3.5" />
                Website
              </a>
            )}
          </div>
        </div>
        {company.summary && (
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground line-clamp-3">
            {company.summary}
          </p>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {/* Key Metrics */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="font-display text-sm">Key Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricRow label="Market Cap" value={formatCurrency(metrics?.market_cap)} />
            <MetricRow label="P/E Ratio" value={metrics?.pe_ratio?.toFixed(2)} />
            <MetricRow label="Forward P/E" value={metrics?.forward_pe?.toFixed(2)} />
            <MetricRow label="Price/Book" value={metrics?.price_to_book?.toFixed(2)} />
            <MetricRow label="Beta" value={metrics?.beta?.toFixed(2)} />
            <MetricRow label="Dividend Yield" value={formatPercent(metrics?.dividend_yield)} />
          </CardContent>
        </Card>

        {/* Price Range */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="font-display text-sm">52-Week Range</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricRow label="52W High" value={formatCurrency(metrics?.["52wk_high"])} />
            <MetricRow label="52W Low" value={formatCurrency(metrics?.["52wk_low"])} />
          </CardContent>
        </Card>

        {/* Technical Snapshot */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="font-display text-sm">Technical Snapshot</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricRow
              label="RSI (14)"
              value={rsi != null ? `${rsi.toFixed(1)} ${rsi > 70 ? "⚠ OB" : rsi < 30 ? "⚠ OS" : ""}` : undefined}
            />
            <MetricRow
              label="MACD Histogram"
              value={macd?.histogram != null ? macd.histogram.toFixed(4) : undefined}
            />
            <MetricRow label="ADX" value={adx?.toFixed(1)} />
            <MetricRow label="BB Position" value={bb?.position} />
          </CardContent>
        </Card>

        {/* Market Regime */}
        {regime && Object.keys(regime).length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="font-display text-sm">Market Regime</CardTitle>
            </CardHeader>
            <CardContent>
              {Object.entries(regime).slice(0, 4).map(([k, v]) => (
                <MetricRow key={k} label={k} value={String(v)} />
              ))}
            </CardContent>
          </Card>
        )}

        {/* Patterns */}
        {patterns?.length > 0 && (
          <Card className="md:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="font-display text-sm">
                Detected Patterns
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {patterns.slice(0, 6).map((p, i) => (
                <PatternBadge key={i} pattern={p} />
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
