export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL ??
  API_BASE.replace(/^http/, "ws");

export const PERIODS = ["1W", "1M", "3M", "6M", "1Y"] as const;
export type Period = (typeof PERIODS)[number];

export const PERIOD_MAP: Record<Period, string> = {
  "1W": "1w",
  "1M": "1mo",
  "3M": "3mo",
  "6M": "6mo",
  "1Y": "1y",
};

export const INDICATORS = ["MA", "BB", "RSI", "MACD"] as const;
export type Indicator = (typeof INDICATORS)[number];

export const PROVIDERS = ["google_genai", "openai", "anthropic"] as const;
export type Provider = (typeof PROVIDERS)[number];

export const PROVIDER_LABELS: Record<Provider, string> = {
  google_genai: "Google Gemini",
  openai: "OpenAI",
  anthropic: "Anthropic",
};
