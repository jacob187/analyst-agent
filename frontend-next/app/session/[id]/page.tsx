import { Archive } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { SessionActions } from "./SessionActions";
import { api } from "@/lib/api";

interface Props {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ ticker?: string }>;
}

export default async function SessionPage({ params, searchParams }: Props) {
  const { id } = await params;
  const { ticker = "" } = await searchParams;

  const { messages } = await api
    .messages(id, ticker)
    .catch(() => ({ messages: [] }));

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col px-4 py-6 sm:px-6">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <SessionActions sessionId={id} ticker={ticker} />
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
      </div>

      {/* Messages */}
      <div className="flex-1 rounded-xl border border-border/60 bg-card overflow-hidden">
        <ScrollArea className="h-full p-4">
          {messages.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No messages in this session.
            </p>
          ) : (
            <div className="space-y-4">
              {messages.map((m, i) => (
                <ChatMessage
                  key={i}
                  message={{ id: String(i), role: m.role, content: m.content }}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}
