import { API_BASE, DEV_USER_ID } from "./constants";
import type {
  ApiKeys,
  BriefingResponse,
  ChartResponse,
  CompanyProfileResponse,
  EnvKeysResponse,
  FilingsResponse,
  FilingStreamEvent,
  Message,
  ModelsResponse,
  QuotesResponse,
  Session,
  TickerSummary,
  WatchlistResponse,
} from "@/types";

// ─── Header helpers ────────────────────────────────────────────────────────────

function keyHeaders(keys: Partial<ApiKeys>): Record<string, string> {
  const h: Record<string, string> = {};
  if (keys.google_api_key) h["X-Google-Api-Key"] = keys.google_api_key;
  if (keys.openai_api_key) h["X-Openai-Api-Key"] = keys.openai_api_key;
  if (keys.anthropic_api_key)
    h["X-Anthropic-Api-Key"] = keys.anthropic_api_key;
  if (keys.tavily_api_key) h["X-Tavily-Api-Key"] = keys.tavily_api_key;
  if (keys.model_id) h["X-Model-Id"] = keys.model_id;
  return h;
}

interface ClerkBrowser {
  user?: { id: string } | null;
  session?: { getToken: () => Promise<string | null> } | null;
}

/**
 * Read the Clerk identity from the browser, if a Clerk session exists: the
 * session JWT plus the signed-in user's id (the token's subject). Returns `{}`
 * when Clerk hasn't loaded yet or the user is signed out. In dev/self-host with
 * Clerk disabled, the backend ignores the missing token.
 */
export async function getClerkIdentity(): Promise<{ token?: string; userId?: string }> {
  if (typeof window === "undefined") return {};
  const clerk = (window as unknown as { Clerk?: ClerkBrowser }).Clerk;
  try {
    const token = await clerk?.session?.getToken();
    // The id only travels with a token: the backend rejects an id it can't verify.
    return token ? { token, userId: clerk?.user?.id } : {};
  } catch {
    return {};
  }
}

/**
 * Identity headers for browser calls. With Clerk enabled the session token is
 * sent together with the signed-in user's id, which the backend requires to
 * equal the token's subject (older backends read only the id). In
 * auth-disabled mode X-User-Id is the configured dev id. Server components pass
 * `serverAuthHeaders()` instead.
 */
async function authHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  if (DEV_USER_ID) headers["X-User-Id"] = DEV_USER_ID;
  const { token, userId } = await getClerkIdentity();
  if (token) {
    headers["X-Clerk-Session-Token"] = token;
    if (userId) headers["X-User-Id"] = userId;
  }
  return headers;
}

async function get<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { ...(await authHeaders()), ...headers },
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function post<T>(
  path: string,
  body: unknown,
  headers?: Record<string, string>
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeaders()), ...headers },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function del<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
    headers: { ...(await authHeaders()), ...headers },
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

// ─── Endpoints ─────────────────────────────────────────────────────────────────

export const api = {
  models: () => get<ModelsResponse>("/models"),

  envKeys: () => get<EnvKeysResponse>("/env-keys"),

  // Chart
  chart: (ticker: string, period = "1y") =>
    get<ChartResponse>(`/stock/${ticker}/chart?period=${period}`),

  quotes: (tickers: string[]) =>
    post<QuotesResponse>("/stock/quotes", { tickers }),

  // Company
  profile: (ticker: string) =>
    get<CompanyProfileResponse>(`/api/company/${ticker}/profile`),

  filings: (ticker: string, keys: ApiKeys) =>
    get<FilingsResponse>(
      `/api/company/${ticker}/filings`,
      keyHeaders(keys)
    ),

  /**
   * Stream filing analysis via Server-Sent Events.
   * Returns an AbortController — call `.abort()` to cancel.
   *
   * Using fetch instead of EventSource because EventSource doesn't support
   * custom request headers (needed for API key forwarding).
   */
  filingsStream(
    ticker: string,
    keys: ApiKeys,
    callbacks: {
      onEvent(event: FilingStreamEvent): void;
      onError(message: string): void;
    }
  ): AbortController {
    const controller = new AbortController();

    (async () => {
      let res: Response;
      try {
        res = await fetch(`${API_BASE}/api/company/${ticker}/filings/stream`, {
          headers: { ...(await authHeaders()), ...keyHeaders(keys) },
          signal: controller.signal,
        });
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          callbacks.onError(String(err));
        }
        return;
      }

      if (!res.ok) {
        callbacks.onError(`API error ${res.status}`);
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          // SSE events are separated by double newlines
          const parts = buffer.split("\n\n");
          buffer = parts.pop() ?? "";

          for (const part of parts) {
            for (const line of part.split("\n")) {
              if (line.startsWith("data: ")) {
                try {
                  callbacks.onEvent(JSON.parse(line.slice(6)) as FilingStreamEvent);
                } catch {
                  // ignore malformed lines
                }
              }
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          callbacks.onError(String(err));
        }
      }
    })();

    return controller;
  },

  // Sessions
  tickers: () =>
    get<{ tickers: TickerSummary[] }>("/tickers"),

  sessions: (ticker: string, headers?: Record<string, string>) =>
    get<{ sessions: Session[] }>(`/sessions?ticker=${ticker}`, headers),

  sessionByTicker: (ticker: string, headers?: Record<string, string>) =>
    get<{ session: Session | null }>(`/sessions/by-ticker/${ticker}`, headers),

  messages: (sessionId: string, ticker: string, headers?: Record<string, string>) =>
    get<{ messages: Message[] }>(
      `/sessions/${sessionId}/messages?ticker=${ticker}`,
      headers
    ),

  deleteSession: (sessionId: string, ticker: string) =>
    del<{ success: boolean }>(
      `/sessions/${sessionId}?ticker=${ticker}`
    ),

  // Watchlist
  watchlist: (headers?: Record<string, string>) =>
    get<WatchlistResponse>("/watchlist", headers),

  addToWatchlist: (ticker: string) =>
    post<{ success: boolean; ticker: string }>("/watchlist", { ticker }),

  removeFromWatchlist: (ticker: string) =>
    del<{ success: boolean }>(`/watchlist/${ticker}`),

  briefing: (keys: ApiKeys) =>
    get<BriefingResponse>("/watchlist/briefing", keyHeaders(keys)),

  briefingHistory: () =>
    get<{ briefings: unknown[] }>("/watchlist/briefing/history"),
};
