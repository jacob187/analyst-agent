"use client";

import { useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SessionActionsProps {
  sessionId: string;
  ticker: string;
}

export function SessionActions({ sessionId, ticker }: SessionActionsProps) {
  const router = useRouter();
  return (
    <div className="flex items-center gap-2">
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 rounded-full"
        onClick={() => router.back()}
      >
        <ArrowLeft className="h-4 w-4" />
      </Button>
      {ticker && (
        <Button
          size="sm"
          className="rounded-full gap-1"
          onClick={() => router.push(`/company/${ticker}?session=${sessionId}`)}
        >
          Continue
          <ArrowRight className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  );
}
