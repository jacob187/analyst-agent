"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight, Archive } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { api } from "@/lib/api";
import type { Message } from "@/types";

export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const ticker = searchParams.get("ticker") ?? "";

  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ticker) return;
    api
      .messages(id, ticker)
      .then((r) => setMessages(r.messages))
      .catch(() => null)
      .finally(() => setLoading(false));
  }, [id, ticker]);

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col px-4 py-6 sm:px-6">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 rounded-full"
            onClick={() => router.back()}
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-bold">{ticker}</span>
              <Badge variant="secondary" className="gap-1 text-xs">
                <Archive className="h-3 w-3" />
                Archived
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">Read-only view</p>
          </div>
        </div>
        {ticker && (
          <Button
            size="sm"
            className="rounded-full gap-1"
            onClick={() => router.push(`/company/${ticker}?session=${id}`)}
          >
            Continue
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 rounded-xl border border-border/60 bg-card overflow-hidden">
        <ScrollArea className="h-full p-4">
          {loading ? (
            <div className="space-y-4">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className={`h-16 rounded-2xl ${i % 2 === 0 ? "ml-auto w-2/3" : "w-3/4"}`} />
              ))}
            </div>
          ) : messages.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No messages in this session.
            </p>
          ) : (
            <div className="space-y-4">
              {messages.map((m, i) => (
                <ChatMessage
                  key={i}
                  message={{
                    id: String(i),
                    role: m.role,
                    content: m.content,
                  }}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}
