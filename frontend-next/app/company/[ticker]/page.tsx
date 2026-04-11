import { CompanyDashboard } from "@/components/company/CompanyDashboard";
import { api } from "@/lib/api";

interface Props {
  params: Promise<{ ticker: string }>;
  searchParams: Promise<{ session?: string }>;
}

export default async function CompanyPage({ params, searchParams }: Props) {
  const { ticker } = await params;
  const { session } = await searchParams;
  const upper = ticker.toUpperCase();

  // Fetch profile and session lookup in parallel — both are cheap reads
  const [profile, sessionRes] = await Promise.all([
    api.profile(upper).catch(() => null),
    !session ? api.sessionByTicker(upper).catch(() => ({ session: null })) : null,
  ]);

  // Use URL-provided session first, fall back to the most recent session for this ticker
  const resolvedSessionId = session ?? sessionRes?.session?.id;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <CompanyDashboard
        ticker={upper}
        initialProfile={profile}
        initialSessionId={resolvedSessionId}
      />
    </div>
  );
}
