import { API_BASE } from "./constants";
import { getUserId } from "@/hooks/useUserId";
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
  if (keys.sec_header) h["X-Sec-Header"] = keys.sec_header;
  if (keys.tavily_api_key) h["X-Tavily-Api-Key"] = keys.tavily_api_key;
  if (keys.model_id) h["X-Model-Id"] = keys.model_id;
  return h;
}

function userIdHeader(): Record<string, string> {
  return { "X-User-Id": getUserId() };
}

async function get<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { ...userIdHeader(), ...headers },
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
    headers: { "Content-Type": "application/json", ...userIdHeader(), ...headers },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function del<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
    headers: { ...userIdHeader(), ...headers },
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
          headers: { ...userIdHeader(), ...keyHeaders(keys) },
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

  sessions: (ticker: string) =>
    get<{ sessions: Session[] }>(`/sessions?ticker=${ticker}`),

  sessionByTicker: (ticker: string) =>
    get<{ session: Session | null }>(`/sessions/by-ticker/${ticker}`),

  messages: (sessionId: string, ticker: string) =>
    get<{ messages: Message[] }>(
      `/sessions/${sessionId}/messages?ticker=${ticker}`
    ),

  deleteSession: (sessionId: string, ticker: string) =>
    del<{ success: boolean }>(
      `/sessions/${sessionId}?ticker=${ticker}`
    ),

  // Watchlist
  watchlist: () => get<WatchlistResponse>("/watchlist"),

  addToWatchlist: (ticker: string) =>
    post<{ success: boolean; ticker: string }>("/watchlist", { ticker }),

  removeFromWatchlist: (ticker: string) =>
    del<{ success: boolean }>(`/watchlist/${ticker}`),

  briefing: (keys: ApiKeys) =>
    get<BriefingResponse>("/watchlist/briefing", keyHeaders(keys)),

  briefingHistory: () =>
    get<{ briefings: unknown[] }>("/watchlist/briefing/history"),
};
