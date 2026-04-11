import { CompanyDashboard } from "@/components/company/CompanyDashboard";

interface Props {
  params: Promise<{ ticker: string }>;
  searchParams: Promise<{ session?: string }>;
}

export default async function CompanyPage({ params, searchParams }: Props) {
  const { ticker } = await params;
  const { session } = await searchParams;
  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <CompanyDashboard
        ticker={ticker.toUpperCase()}
        initialSessionId={session}
      />
    </div>
  );
}
