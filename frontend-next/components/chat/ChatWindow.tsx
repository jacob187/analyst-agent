"use client";

import { useEffect, useRef, useState } from "react";
import { Send, Wifi, WifiOff, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "./ChatMessage";
import { useWebSocket } from "@/hooks/useWebSocket";
import { cn } from "@/lib/utils";
import type { ApiKeys } from "@/types";

interface ChatWindowProps {
  ticker: string;
  keys: ApiKeys;
  initialSessionId?: string;
}

export function ChatWindow({ ticker, keys, initialSessionId }: ChatWindowProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // History loading is handled inside the hook — no callback needed here
  const { messages, status, sendMessage } = useWebSocket({
    ticker,
    keys,
    sessionId: initialSessionId,
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    const text = input.trim();
    if (!text || status !== "connected") return;
    sendMessage(text);
    setInput("");
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const statusColor = {
    connecting: "text-yellow-500",
    connected: "text-primary",
    disconnected: "text-muted-foreground",
    error: "text-destructive",
  }[status];

  const StatusIcon =
    status === "connecting" ? Loader2 : status === "connected" ? Wifi : WifiOff;

  return (
    <div className="flex h-full flex-col rounded-xl border border-border/60 bg-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="font-display text-sm font-semibold">{ticker}</span>
          <span className="text-xs text-muted-foreground">AI Analyst</span>
        </div>
        <div className={cn("flex items-center gap-1.5 text-xs", statusColor)}>
          <StatusIcon
            className={cn("h-3.5 w-3.5", status === "connecting" && "animate-spin")}
          />
          <span className="capitalize">{status}</span>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center py-12 text-center">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
              <Send className="h-5 w-5 text-primary" />
            </div>
            <p className="text-sm font-medium">Ask anything about {ticker}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Try: &quot;What are the main risks?&quot; or &quot;Summarize the latest 10-K&quot;
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border/60 p-3">
        <div className="flex items-end gap-2 rounded-xl border border-border/60 bg-background px-3 py-2 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              status === "connected"
                ? "Ask about this company…"
                : status === "connecting"
                ? "Connecting…"
                : "Disconnected"
            }
            disabled={status !== "connected"}
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
            style={{ maxHeight: "120px" }}
          />
          <Button
            size="icon"
            className="h-7 w-7 shrink-0 rounded-lg"
            disabled={!input.trim() || status !== "connected"}
            onClick={handleSend}
          >
            <Send className="h-3.5 w-3.5" />
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
