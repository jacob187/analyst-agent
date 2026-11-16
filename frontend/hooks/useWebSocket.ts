"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_BASE, DEV_USER_ID } from "@/lib/constants";
import { api, getClerkIdentity } from "@/lib/api";
import { useAuth } from "@/lib/auth";
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

  // Identity gate: the socket must not open until Clerk has resolved, or it
  // would authenticate anonymously for its whole life (chat never saved).
  // `userId` is in connect's deps so a sign-in/out reconnects with the new identity.
  const { userId, isLoaded: loaded } = useAuth();

  const wsRef = useRef<WebSocket | null>(null);
  // Identity the transcript on screen belongs to. When it changes, the new
  // server session knows nothing of that transcript, so connect() drops it.
  const identityRef = useRef<string | null | undefined>(undefined);
  // The session id prop was resolved for the previous identity once this is set.
  const dropSessionIdRef = useRef(false);
  // Id of the assistant bubble currently being filled. Set and cleared
  // synchronously in the WS handler (never inside a setMessages updater): React
  // runs updaters lazily, so a ref mutated or read there can be stale by the
  // time it runs — which split one reply across two bubbles.
  const streamingIdRef = useRef<string | null>(null);

  // ── Refs for unstable values ────────────────────────────────────────────────
  // Reading from refs inside connect() means we never need them as deps,
  // so the effect only fires once per ticker change (not on every render).
  const keysRef = useRef(keys);
  const sessionIdRef = useRef(sessionId);
  useEffect(() => {
    keysRef.current = keys;
    sessionIdRef.current = sessionId;
  });

  // ── connect ─────────────────────────────────────────────────────────────────
  // Depends on `ticker` and the resolved identity; keys/sessionId come from refs.
  const connect = useCallback(() => {
    if (!loaded) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const identity = userId ?? null;
    if (identityRef.current !== undefined && identityRef.current !== identity) {
      // Signed in/out mid-chat: the superseded socket's in-flight bubble will
      // never complete and its transcript isn't the new session's history.
      streamingIdRef.current = null;
      dropSessionIdRef.current = true;
      setMessages([]);
      setActiveSessionId(null);
    }
    identityRef.current = identity;

    setStatus("connecting");
    const ws = new WebSocket(`${WS_BASE}/ws/chat/${ticker}`);
    wsRef.current = ws;

    ws.onopen = async () => {
      const k = keysRef.current;
      const clerk = await getClerkIdentity();
      // Superseded while awaiting the token (ticker/identity change).
      if (wsRef.current !== ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(
        JSON.stringify({
          type: "auth",
          // Clerk token plus the user id it was issued to (the backend rejects
          // an id that differs from the token's subject; older backends read
          // only the id). DEV_USER_ID stands in when auth is disabled.
          user_id: DEV_USER_ID ?? clerk.userId,
          clerk_session_token: clerk.token,
          google_api_key: k.google_api_key || undefined,
          openai_api_key: k.openai_api_key || undefined,
          anthropic_api_key: k.anthropic_api_key || undefined,
          tavily_api_key: k.tavily_api_key || undefined,
          model_id: k.model_id || undefined,
          session_id: dropSessionIdRef.current
            ? undefined
            : sessionIdRef.current || undefined,
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
              api
                .messages(data.session_id, ticker)
                .then((body) => {
                  // The socket was superseded (identity change) while fetching.
                  if (wsRef.current !== ws) return;
                  setMessages(
                    body.messages.map((m) => ({
                      id: crypto.randomUUID(),
                      role: m.role,
                      content: m.content,
                    }))
                  );
                })
                .catch(() => {
                  // Non-2xx (or malformed body): say so rather than show a blank chat.
                  setMessages((prev) => [
                    ...prev,
                    {
                      id: crypto.randomUUID(),
                      role: "assistant",
                      content: "Couldn't load previous messages.",
                      isStreaming: false,
                    },
                  ]);
                });
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
      // A superseded socket (ticker/identity change) closes after its
      // replacement is already open; don't clobber the new socket's state.
      if (wsRef.current !== ws) return;
      setStatus("disconnected");
      streamingIdRef.current = null;
    };

    ws.onerror = () => {
      if (wsRef.current === ws) setStatus("error");
    };

  }, [ticker, loaded, userId]); // keys/sessionId read from refs at call time

  useEffect(() => {
    // Opens the external WebSocket and mirrors its status into state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
