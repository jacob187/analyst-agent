"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { exportSessionAsMarkdown } from "@/lib/export";

interface SessionActionsProps {
  sessionId: string;
  ticker: string;
  createdAt: string;
  model: string | null;
}

export function SessionActions({ sessionId, ticker, createdAt, model }: SessionActionsProps) {
  const router = useRouter();
  const [exporting, setExporting] = useState(false);

  async function handleExport() {
    setExporting(true);
    try {
      const { messages } = await api.messages(sessionId, ticker);
      exportSessionAsMarkdown(ticker, createdAt, model, messages);
    } finally {
      setExporting(false);
    }
  }

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
        <>
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full gap-1 text-muted-foreground"
            onClick={handleExport}
            disabled={exporting}
          >
            <Download className="h-3.5 w-3.5" />
            {exporting ? "Exporting…" : "Export"}
          </Button>
          <Button
            size="sm"
            className="rounded-full gap-1"
            onClick={() => router.push(`/company/${ticker}?session=${sessionId}`)}
          >
            Continue
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </>
      )}
    </div>
  );
}
