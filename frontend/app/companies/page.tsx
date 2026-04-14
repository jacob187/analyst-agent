"use client";

import { useEffect, useState } from "react";
import { Building2, Loader2 } from "lucide-react";
import { CompanyCard } from "@/components/companies/CompanyCard";
import { api } from "@/lib/api";
import type { TickerSummary } from "@/types";

export default function CompaniesPage() {
  const [tickers, setTickers] = useState<TickerSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .tickers()
      .then(({ tickers }) => setTickers(tickers))
      .catch(() => setTickers([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">Saved Companies</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Companies you've analyzed.
        </p>
      </div>

      {loading ? (
        <div className="py-16 text-center">
          <Loader2 className="mx-auto mb-3 h-10 w-10 animate-spin text-muted-foreground/40" />
        </div>
      ) : tickers.length === 0 ? (
        <div className="py-16 text-center">
          <Building2 className="mx-auto mb-3 h-10 w-10 text-muted-foreground/40" />
          <p className="text-sm text-muted-foreground">No companies yet.</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {tickers.map((t) => (
            <CompanyCard key={t.ticker} ticker={t} />
          ))}
        </div>
      )}
    </div>
  );
}
