import { API_BASE } from "./constants";
import type {
  ApiKeys,
  BriefingResponse,
  ChartResponse,
  CompanyProfileResponse,
  EnvKeysResponse,
  FilingsResponse,
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

async function get<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers });
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
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
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
