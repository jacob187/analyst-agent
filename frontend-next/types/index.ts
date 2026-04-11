// ─── LLM Models ────────────────────────────────────────────────────────────────

export interface Model {
  id: string;
  provider: "google_genai" | "openai" | "anthropic";
  display_name: string;
  max_context: number;
  thinking_capable: boolean;
  default: boolean;
}

export interface ModelsResponse {
  models: Model[];
}

export interface EnvKeysResponse {
  google: boolean;
  openai: boolean;
  anthropic: boolean;
  sec_header: boolean;
  tavily: boolean;
}

// ─── API Keys ──────────────────────────────────────────────────────────────────

export interface ApiKeys {
  google_api_key: string;
  openai_api_key: string;
  anthropic_api_key: string;
  sec_header: string;
  tavily_api_key: string;
  model_id: string;
}

// ─── Sessions ──────────────────────────────────────────────────────────────────

export interface Session {
  id: string;
  ticker: string;
  model: string | null;
  created_at: string;
}

export interface TickerSummary {
  ticker: string;
  session_count: number;
  last_active: string;
}

export interface Message {
  id?: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
}

// ─── Stock / Chart ─────────────────────────────────────────────────────────────

export interface Candle {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Pattern {
  type: string;
  direction: "bullish" | "bearish";
  confidence: number;
  time?: string | number;
  status?: string;
}

export interface ChartResponse {
  candles: Candle[];
  indicators: Record<string, unknown>;
  quote: { price: number; currency: string };
  patterns: Pattern[];
}

export interface Quote {
  price: number;
  previousClose: number;
  change: number;
  changePercent: number;
}

export interface QuotesResponse {
  quotes: Record<string, Quote>;
}

// ─── Company Profile ───────────────────────────────────────────────────────────

export interface CompanyInfo {
  name: string;
  sector: string;
  industry: string;
  country: string;
  website: string;
  summary: string;
  employees: number;
}

export interface Metrics {
  market_cap: number;
  pe_ratio: number;
  forward_pe: number;
  price_to_book: number;
  "52wk_high": number;
  "52wk_low": number;
  dividend_yield: number;
  beta: number;
}

export interface CompanyProfileResponse {
  ticker: string;
  company: CompanyInfo;
  metrics: Metrics;
  quote: Quote;
  earnings: unknown[];
  technicals: Record<string, unknown>;
  patterns: Pattern[];
  regime: Record<string, unknown>;
}

// ─── Filings ───────────────────────────────────────────────────────────────────

export interface FilingMetadata {
  cik: string;
  accession: string;
  filing_date: string;
  period_of_report: string;
  edgar_url: string;
  form_type?: string;
}

export interface FilingAnalysis {
  [key: string]: unknown;
}

export interface FilingsResponse {
  ticker: string;
  tenk?: {
    metadata: FilingMetadata;
    risk_10k?: FilingAnalysis;
    mda_10k?: FilingAnalysis;
    balance?: FilingAnalysis;
  };
  tenq?: {
    metadata: FilingMetadata;
    risk_10q?: FilingAnalysis;
    mda_10q?: FilingAnalysis;
  };
  earnings?: {
    has_earnings: boolean;
    metadata?: FilingMetadata;
    analysis?: FilingAnalysis;
  };
}

// ─── Watchlist ─────────────────────────────────────────────────────────────────

export interface WatchlistItem {
  ticker: string;
  added_at: string;
  name: string | null;
  sector: string | null;
}

export interface WatchlistResponse {
  tickers: WatchlistItem[];
}

export interface BriefingResponse {
  briefing: string;
  thinking?: string;
  structured?: Record<string, unknown>;
  tickers: string[];
}

// ─── Filing SSE Stream ─────────────────────────────────────────────────────────

export interface FilingProgressEvent {
  type: "progress";
  step: string; // "edgar_fetch" | "10-K/risk_10k" | ...
  status: "fetching" | "complete" | "processing" | "cached" | "done" | "failed";
  duration?: number;
}

export interface FilingMetadataEvent {
  type: "metadata";
  tenk_metadata: FilingMetadata | null;
  tenq_metadata: FilingMetadata | null;
  earnings_has_earnings: boolean;
  earnings_metadata: FilingMetadata | null;
}

export interface FilingSectionEvent {
  type: "section";
  form: string; // "10-K" | "10-Q" | "8-K"
  key: string;  // "risk_10k" | "mda_10k" | "balance" | "risk_10q" | "mda_10q" | "earnings"
  data: FilingAnalysis;
}

export interface FilingCompleteEvent {
  type: "complete";
}

export interface FilingErrorEvent {
  type: "error";
  message: string;
}

export type FilingStreamEvent =
  | FilingProgressEvent
  | FilingMetadataEvent
  | FilingSectionEvent
  | FilingCompleteEvent
  | FilingErrorEvent;

// ─── WebSocket Messages ────────────────────────────────────────────────────────

export type WsMessageType =
  | "auth_success"
  | "system"
  | "node"
  | "tool"
  | "thinking"
  | "token"
  | "response"
  | "error";

export interface WsMessage {
  type: WsMessageType;
  message?: string;
  session_id?: string;
  resumed?: boolean;
  model_id?: string;
  tool?: string;
  step?: number;
  total?: number;
}

// ─── UI State ──────────────────────────────────────────────────────────────────

export interface StreamingMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  isStreaming?: boolean;
  status?: string;
}
