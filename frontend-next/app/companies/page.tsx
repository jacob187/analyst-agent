"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Building2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { CompanyCard } from "@/components/companies/CompanyCard";
import { api } from "@/lib/api";
import type { TickerSummary } from "@/types";

export default function CompaniesPage() {
  const router = useRouter();
  const [tickers, setTickers] = useState<TickerSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .tickers()
      .then((r) => setTickers(r.tickers))
      .catch(() => null)
      .finally(() => setLoading(false));
  }, []);

  async function handleOpen(ticker: string) {
    const res = await api.sessionByTicker(ticker).catch(() => ({ session: null }));
    const sessionId = res.session?.id;
    const url = sessionId
      ? `/company/${ticker}?session=${sessionId}`
      : `/company/${ticker}`;
    router.push(url);
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">
          Saved Companies
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Companies with existing analysis sessions.
        </p>
      </div>

      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {[...Array(6)].map((_, i) => (
            <Skeleton key={i} className="h-36 rounded-xl" />
          ))}
        </div>
      ) : tickers.length === 0 ? (
        <div className="py-16 text-center">
          <Building2 className="mx-auto mb-3 h-10 w-10 text-muted-foreground/40" />
          <p className="text-sm text-muted-foreground">No companies yet.</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {tickers.map((t) => (
            <CompanyCard key={t.ticker} ticker={t} onOpen={handleOpen} />
          ))}
        </div>
      )}
    </div>
  );
}
