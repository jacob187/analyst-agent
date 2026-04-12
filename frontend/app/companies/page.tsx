import { Building2 } from "lucide-react";
import { CompanyCard } from "@/components/companies/CompanyCard";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function CompaniesPage() {
  const { tickers } = await api.tickers().catch(() => ({ tickers: [] }));

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">Saved Companies</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Companies you've analyzed.
        </p>
      </div>

      {tickers.length === 0 ? (
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
