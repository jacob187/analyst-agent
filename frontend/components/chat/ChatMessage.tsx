"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { marked } from "marked";
import DOMPurify from "isomorphic-dompurify";
import { cn } from "@/lib/utils";
import { useElapsedSeconds } from "@/hooks/useElapsedSeconds";
import type { StreamingMessage } from "@/types";
import { latestThinkingPreview } from "./thinkingPreview";

marked.setOptions({ breaks: true });

interface ChatMessageProps {
  message: StreamingMessage;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const isUser = message.role === "user";
  // Progress UI (status pill, elapsed counter, live thinking) only before answer tokens arrive
  const showProgress = !isUser && !!message.isStreaming && !message.content;
  const elapsed = useElapsedSeconds(showProgress);
  const thinkingPreview = showProgress ? latestThinkingPreview(message.thinking) : "";

  // Safely inject sanitized markdown HTML via DOM ref (avoids dangerouslySetInnerHTML)
  useEffect(() => {
    if (!contentRef.current || !message.content || isUser) return;
    const raw = marked.parse(message.content) as string;
    // DOMPurify strips all XSS vectors before injection
    contentRef.current.innerHTML = DOMPurify.sanitize(raw);
  }, [message.content, isUser]);

  return (
    <div
      className={cn(
        "flex w-full gap-3 animate-fade-in-up",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20 text-[10px] font-bold text-primary">
          AI
        </div>
      )}

      <div className={cn("max-w-[80%] space-y-2", isUser ? "items-end" : "items-start")}>
        {/* Collapsible thinking block */}
        {message.thinking && (
          <div className="rounded-lg border border-border/40 bg-muted/30 text-xs">
            <button
              onClick={() => setThinkingOpen((o) => !o)}
              className="flex w-full items-center justify-between px-3 py-2 text-muted-foreground hover:text-foreground"
            >
              <span className="font-medium">Thinking</span>
              <ChevronDown
                className={cn("h-3.5 w-3.5 transition-transform", thinkingOpen && "rotate-180")}
              />
            </button>
            {thinkingOpen && (
              <div className="border-t border-border/40 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap">
                {message.thinking}
              </div>
            )}
          </div>
        )}

        {/* Streaming status pill */}
        {showProgress && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 empty:hidden">
              {message.status && (
                <div className="flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
                  <span className="streaming-dot" />
                  {message.status}
                </div>
              )}
              {elapsed >= 1 && (
                <span
                  className="text-xs tabular-nums text-muted-foreground/70"
                  aria-label={`${elapsed} seconds elapsed`}
                >
                  {elapsed}s
                </span>
              )}
            </div>
            {/* Live region stays mounted while streaming so updates are announced */}
            <div aria-live="polite" className="empty:hidden">
              {thinkingPreview && (
                <p className="line-clamp-2 whitespace-pre-line px-1 text-xs italic leading-relaxed text-muted-foreground/70">
                  {thinkingPreview}
                  <span
                    aria-hidden="true"
                    className="ml-0.5 inline-block motion-safe:animate-pulse"
                  >
                    …
                  </span>
                </p>
              )}
            </div>
          </div>
        )}

        {/* Message bubble */}
        {(message.content || isUser) && (
          <div
            className={cn(
              "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
              isUser
                ? "bg-primary text-primary-foreground rounded-tr-sm"
                : "border border-border/60 bg-card rounded-tl-sm"
            )}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <div ref={contentRef} className="prose-chat" />
            )}
            {message.isStreaming && message.content && (
              <span className="streaming-dot" />
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-secondary text-[10px] font-bold">
          U
        </div>
      )}
    </div>
  );
}
