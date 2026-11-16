"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { WS_BASE, API_BASE } from "@/lib/constants";
import { getClerkToken } from "@/lib/api";
import { getUserId } from "@/hooks/useUserId";
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
  const [freeTrial, setFreeTrial] = useState<{ queries: number } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  // Id of the assistant bubble currently being filled. Set and cleared
  // synchronously in the WS handler (never inside a setMessages updater): React
  // runs updaters lazily, so a ref mutated or read there can be stale by the
  // time it runs — which split one reply across two bubbles.
  const streamingIdRef = useRef<string | null>(null);

  // ── Refs for unstable values ────────────────────────────────────────────────
  // Reading from refs inside connect() means we never need them as deps,
  // so the effect only fires once per ticker change (not on every render).
  // Synced in a layout effect (not during render), which runs before the
  // passive effects below, so connect() still sees the current values.
  const keysRef = useRef(keys);
  const sessionIdRef = useRef(sessionId);
  useLayoutEffect(() => {
    keysRef.current = keys;
    sessionIdRef.current = sessionId;
  });

  // ── connect ─────────────────────────────────────────────────────────────────
  // Only depends on `ticker` — a stable string from the route param.
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(`${WS_BASE}/ws/chat/${ticker}`);
    wsRef.current = ws;

    ws.onopen = async () => {
      const k = keysRef.current;
      const clerkToken = await getClerkToken();
      ws.send(
        JSON.stringify({
          type: "auth",
          user_id: getUserId() || undefined,
          clerk_session_token: clerkToken || undefined,
          google_api_key: k.google_api_key || undefined,
          openai_api_key: k.openai_api_key || undefined,
          anthropic_api_key: k.anthropic_api_key || undefined,
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
          setFreeTrial(
            data.free_trial ? { queries: data.free_trial_queries ?? 0 } : null
          );
          if (data.session_id) {
            setActiveSessionId(data.session_id);
            // If resuming, fetch and display prior messages
            if (data.resumed) {
              getClerkToken().then((tok) => {
                const id = getUserId();
                const headers: Record<string, string> = {};
                if (id) headers["X-User-Id"] = id;
                if (tok) headers["X-Clerk-Session-Token"] = tok;
                return fetch(
                  `${API_BASE}/sessions/${data.session_id}/messages?ticker=${ticker}`,
                  { headers }
                );
              })
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

        case "thinking": {
          const id = streamingIdRef.current;
          if (id === null) {
            const newId = crypto.randomUUID();
            streamingIdRef.current = newId;
            setMessages((prev) => [
              ...prev,
              {
                id: newId,
                role: "assistant",
                content: "",
                thinking: data.message ?? "",
                isStreaming: true,
                status: "Thinking…",
              },
            ]);
          } else {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === id
                  ? { ...m, thinking: (m.thinking ?? "") + (data.message ?? "") }
                  : m
              )
            );
          }
          break;
        }

        case "token":
        case "response": {
          const id = streamingIdRef.current;
          const isToken = data.type === "token";
          if (id === null) {
            const newId = crypto.randomUUID();
            // A final response closes the cycle; a token opens one.
            streamingIdRef.current = isToken ? newId : null;
            setMessages((prev) => [
              ...prev,
              {
                id: newId,
                role: "assistant",
                content: data.message ?? "",
                isStreaming: isToken,
                status: isToken ? "Responding…" : undefined,
              },
            ]);
          } else {
            if (!isToken) streamingIdRef.current = null;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === id
                  ? {
                      ...m,
                      content: isToken
                        ? m.content + (data.message ?? "")
                        : (data.message ?? ""),
                      isStreaming: isToken,
                      status: isToken ? "Responding…" : undefined,
                    }
                  : m
              )
            );
          }
          if (!isToken) {
            // Clean up any leftover status-only messages (e.g. "Processing query…")
            // that were created by earlier node/tool events for this response cycle.
            setMessages((prev) =>
              prev.filter((m) => !(m.isStreaming && !m.content))
            );
          }
          break;
        }

        case "node": {
          const id = streamingIdRef.current;
          if (id === null) {
            const newId = crypto.randomUUID();
            streamingIdRef.current = newId;
            setMessages((prev) => [
              ...prev,
              {
                id: newId,
                role: "assistant",
                content: "",
                isStreaming: true,
                status: data.message ?? "Processing…",
              },
            ]);
          } else {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === id ? { ...m, status: data.message ?? m.status } : m
              )
            );
          }
          break;
        }

        case "tool": {
          const id = streamingIdRef.current;
          if (id === null) break;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === id
                ? {
                    ...m,
                    status: data.message ?? `Running ${data.tool ?? "tool"}…`,
                  }
                : m
            )
          );
          break;
        }

        case "error": {
          const id = streamingIdRef.current;
          streamingIdRef.current = null;
          const errorText = `Error: ${data.message ?? "Unknown error"}`;
          if (id === null) {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: errorText,
                isStreaming: false,
              },
            ]);
          } else {
            // Close the in-flight bubble in place: keep any partial answer and
            // append the error, instead of leaving it spinning beside a new one.
            setMessages((prev) =>
              prev.map((m) =>
                m.id === id
                  ? {
                      ...m,
                      content: m.content ? `${m.content}\n\n${errorText}` : errorText,
                      isStreaming: false,
                      status: undefined,
                    }
                  : m
              )
            );
          }
          break;
        }

        default:
          break;
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      streamingIdRef.current = null;
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
    streamingIdRef.current = null;
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: text },
    ]);
    wsRef.current.send(JSON.stringify({ type: "query", message: text }));
  }, []);

  return { messages, status, activeSessionId, sendMessage, freeTrial };
}
