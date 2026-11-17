import { WatchlistPanel } from "@/components/watchlist/WatchlistPanel";
import { api } from "@/lib/api";
import { serverAuthHeaders } from "@/lib/server-auth";
import type { Quote } from "@/types";

export const dynamic = "force-dynamic";

export default async function WatchlistPage() {
  // The browser helpers can't see Clerk here; forward the request's session.
  const authHeaders = await serverAuthHeaders();
  const { tickers } = await api
    .watchlist(authHeaders)
    .catch(() => ({ tickers: [] }));

  // Fetch quotes for all watchlist tickers in the same server request
  const initialQuotes: Record<string, Quote> =
    tickers.length > 0
      ? await api
          .quotes(tickers.map((t) => t.ticker))
          .then((r) => r.quotes)
          .catch(() => ({}))
      : {};

  return (
    <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">Watchlist</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Track up to 10 tickers and generate AI market briefings.
        </p>
      </div>
      <WatchlistPanel initialItems={tickers} initialQuotes={initialQuotes} />
    </div>
  );
}
