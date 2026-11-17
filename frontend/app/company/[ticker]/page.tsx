import { CompanyDashboard } from "@/components/company/CompanyDashboard";
import { api } from "@/lib/api";
import { serverAuthHeaders } from "@/lib/server-auth";


interface Props {
  params: Promise<{ ticker: string }>;
  searchParams: Promise<{ session?: string }>;
}

export default async function CompanyPage({ params, searchParams }: Props) {
  const { ticker } = await params;
  const { session } = await searchParams;
  const upper = ticker.toUpperCase();

  // Only resolve the session server-side (fast SQLite read).
  // Profile is fetched client-side to avoid blocking the page on a slow yfinance call.
  // Signed-out visitors have no per-user session (the lookup would 422), so
  // skip the call when there are no identity headers.
  let resolvedSessionId = session;
  if (!resolvedSessionId) {
    const authHeaders = await serverAuthHeaders();
    if (Object.keys(authHeaders).length > 0) {
      const sessionRes = await api
        .sessionByTicker(upper, authHeaders)
        .catch((err) => {
          console.error("company page: failed to resolve session", err);
          return { session: null };
        });
      resolvedSessionId = sessionRes.session?.id;
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <CompanyDashboard
        ticker={upper}
        initialSessionId={resolvedSessionId}
      />
    </div>
  );
}
