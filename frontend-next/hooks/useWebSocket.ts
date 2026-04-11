"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_BASE, API_BASE } from "@/lib/constants";
import type { ApiKeys, StreamingMessage, WsMessage } from "@/types";

export type WsStatus = "connecting" | "connected" | "disconnected" | "error";

interface UseWebSocketOptions {
  ticker: string;
  keys: ApiKeys;
  sessionId?: string;
}

export function useWebSocket({ ticker, keys, sessionId }: UseWebSocketOptions) {
  const [messages, setMessages] = useState<StreamingMessage[]>([]);
  const [status, setStatus] = useState<WsStatus>("disconnected");
  const [activeSessionId, setActiveSessionId] = useState<string | null>(
    sessionId ?? null
  );

  const wsRef = useRef<WebSocket | null>(null);
  const streamingIdxRef = useRef<number | null>(null);

  // ── Refs for unstable values ────────────────────────────────────────────────
  // Reading from refs inside connect() means we never need them as deps,
  // so the effect only fires once per ticker change (not on every render).
  const keysRef = useRef(keys);
  const sessionIdRef = useRef(sessionId);
  keysRef.current = keys;
  sessionIdRef.current = sessionId;

  // ── connect ─────────────────────────────────────────────────────────────────
  // Only depends on `ticker` — a stable string from the route param.
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(`${WS_BASE}/ws/chat/${ticker}`);
    wsRef.current = ws;

    ws.onopen = () => {
      const k = keysRef.current;
      ws.send(
        JSON.stringify({
          type: "auth",
          google_api_key: k.google_api_key || undefined,
          openai_api_key: k.openai_api_key || undefined,
          anthropic_api_key: k.anthropic_api_key || undefined,
          sec_header: k.sec_header || undefined,
          tavily_api_key: k.tavily_api_key || undefined,
          model_id: k.model_id || undefined,
          session_id: sessionIdRef.current || undefined,
        })
      );
    };

    ws.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data as string) as WsMessage;

      switch (data.type) {
        case "auth_success":
          setStatus("connected");
          if (data.session_id) {
            setActiveSessionId(data.session_id);
            // If resuming, fetch and display prior messages
            if (data.resumed) {
              fetch(
                `${API_BASE}/sessions/${data.session_id}/messages?ticker=${ticker}`
              )
                .then((r) => r.json())
                .then((body: { messages: Array<{ role: "user" | "assistant"; content: string }> }) => {
                  setMessages(
                    body.messages.map((m) => ({
                      id: crypto.randomUUID(),
                      role: m.role,
                      content: m.content,
                    }))
                  );
                })
                .catch(() => null);
            }
          }
          break;

        case "thinking":
          setMessages((prev) => {
            if (streamingIdxRef.current === null) {
              streamingIdxRef.current = prev.length;
              return [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: "",
                  thinking: data.message ?? "",
                  isStreaming: true,
                  status: "Thinking…",
                },
              ];
            }
            return prev.map((m, i) =>
              i === streamingIdxRef.current
                ? { ...m, thinking: (m.thinking ?? "") + (data.message ?? "") }
                : m
            );
          });
          break;

        case "token":
        case "response":
          setMessages((prev) => {
            if (streamingIdxRef.current === null) {
              streamingIdxRef.current = prev.length;
              return [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: data.message ?? "",
                  isStreaming: data.type === "token",
                  status: data.type === "token" ? "Responding…" : undefined,
                },
              ];
            }
            return prev.map((m, i) =>
              i === streamingIdxRef.current
                ? {
                    ...m,
                    content:
                      data.type === "response"
                        ? (data.message ?? "")
                        : m.content + (data.message ?? ""),
                    isStreaming: data.type === "token",
                    status: data.type === "token" ? "Responding…" : undefined,
                  }
                : m
            );
          });
          if (data.type === "response") {
            streamingIdxRef.current = null;
            // Clean up any leftover status-only messages (e.g. "Processing query…")
            // that were created by earlier node/tool events for this response cycle.
            setMessages((prev) =>
              prev.filter((m) => !(m.isStreaming && !m.content))
            );
          }
          break;

        case "node":
          setMessages((prev) => {
            if (streamingIdxRef.current === null) {
              streamingIdxRef.current = prev.length;
              return [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: "",
                  isStreaming: true,
                  status: data.message ?? "Processing…",
                },
              ];
            }
            return prev.map((m, i) =>
              i === streamingIdxRef.current
                ? { ...m, status: data.message ?? m.status }
                : m
            );
          });
          break;

        case "tool":
          setMessages((prev) =>
            prev.map((m, i) =>
              i === streamingIdxRef.current
                ? {
                    ...m,
                    status: data.message ?? `Running ${data.tool ?? "tool"}…`,
                  }
                : m
            )
          );
          break;

        case "error":
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: `Error: ${data.message ?? "Unknown error"}`,
              isStreaming: false,
            },
          ]);
          streamingIdxRef.current = null;
          break;

        default:
          break;
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      streamingIdxRef.current = null;
    };

    ws.onerror = () => setStatus("error");

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]); // ← only ticker; keys/sessionId read from refs at call time

  useEffect(() => {
    connect();

    const handleVisibility = () => {
      if (!document.hidden && wsRef.current?.readyState !== WebSocket.OPEN) {
        connect();
      }
    };

    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibility);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((text: string) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    streamingIdxRef.current = null;
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: text },
    ]);
    wsRef.current.send(JSON.stringify({ type: "query", message: text }));
  }, []);

  return { messages, status, activeSessionId, sendMessage };
}
