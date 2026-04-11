import { WatchlistPanel } from "@/components/watchlist/WatchlistPanel";

export default function WatchlistPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">Watchlist</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Track up to 10 tickers and generate AI market briefings.
        </p>
      </div>
      <WatchlistPanel />
    </div>
  );
}
