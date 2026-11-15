import Link from "next/link";
import { AlertTriangle, Archive } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { SessionActions } from "./SessionActions";
import { api } from "@/lib/api";
import { serverAuthHeaders } from "@/lib/server-auth";

interface Props {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ ticker?: string }>;
}

/** Short, user-facing reason for a failed load, derived from the API error. */
function failureReason(err: unknown): string {
  const status = Number(/API error (\d+)/.exec(String(err))?.[1]);
  if (!status) return "We couldn't reach the server. Check your connection and try again.";
  if (status === 404) return "This conversation wasn't found. It may have been deleted.";
  if (status === 401 || status === 403 || status === 422) {
    return "You need to be signed in to view this conversation.";
  }
  return "The server ran into a problem loading this conversation. Try again shortly.";
}

export default async function SessionPage({ params, searchParams }: Props) {
  const { id } = await params;
  const { ticker = "" } = await searchParams;

  const authHeaders = await serverAuthHeaders();
  const [messagesResult, { sessions }] = await Promise.all([
    api
      .messages(id, ticker, authHeaders)
      .then((data) => ({ ok: true as const, messages: data.messages }))
      .catch((err) => {
        console.error("session page: failed to load messages", err);
        return { ok: false as const, reason: failureReason(err) };
      }),
    api.sessions(ticker, authHeaders).catch((err) => {
      console.error("session page: failed to load sessions", err);
      return { sessions: [] };
    }),
  ]);

  if (!messagesResult.ok) {
    return (
      <div className="mx-auto flex max-w-3xl flex-col px-4 py-6 sm:px-6">
        <div
          role="alert"
          className="flex flex-col items-center gap-3 rounded-xl border border-border/60 bg-card px-4 py-12 text-center"
        >
          <AlertTriangle className="h-6 w-6 text-destructive" />
          <h1 className="font-display font-bold">Couldn&apos;t load this conversation</h1>
          <p className="text-sm text-muted-foreground">{messagesResult.reason}</p>
          <Button asChild variant="outline" size="sm" className="mt-2 rounded-full">
            <Link href="/history">Back to History</Link>
          </Button>
        </div>
      </div>
    );
  }

  const { messages } = messagesResult;
  const session = sessions.find((s) => s.id === id);

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col px-4 py-6 sm:px-6">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <SessionActions
            sessionId={id}
            ticker={ticker}
            createdAt={session?.created_at ?? new Date().toISOString()}
            model={session?.model ?? null}
          />
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
