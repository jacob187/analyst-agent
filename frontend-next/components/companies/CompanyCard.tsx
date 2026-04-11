"use client";

import { useRouter } from "next/navigation";
import { BarChart2, MessageSquare, Clock } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDate } from "@/lib/utils";
import type { TickerSummary } from "@/types";

interface CompanyCardProps {
  ticker: TickerSummary;
  onOpen: (ticker: string) => void;
}

export function CompanyCard({ ticker, onOpen }: CompanyCardProps) {
  return (
    <Card
      className="group cursor-pointer transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
      onClick={() => onOpen(ticker.ticker)}
    >
      <CardContent className="p-5">
        <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 transition-colors group-hover:bg-primary/20">
          <BarChart2 className="h-5 w-5 text-primary" />
        </div>
        <p className="font-display text-xl font-bold">{ticker.ticker}</p>
        <div className="mt-2 flex flex-wrap gap-2">
          <Badge variant="secondary" className="gap-1 text-xs">
            <MessageSquare className="h-2.5 w-2.5" />
            {ticker.session_count} session{ticker.session_count !== 1 ? "s" : ""}
          </Badge>
          <Badge variant="outline" className="gap-1 text-xs">
            <Clock className="h-2.5 w-2.5" />
            {formatDate(ticker.last_active)}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
