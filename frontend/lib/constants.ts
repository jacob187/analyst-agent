export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** True for self-hosters running without Clerk (pair with DISABLE_AUTH=true on the backend). */
export const AUTH_DISABLED = process.env.NEXT_PUBLIC_DISABLE_AUTH === "true";

/** Stable user_id used when auth is disabled — matches the backend's USER_ID_RE. */
export const LOCAL_DEV_USER_ID = "user_localdev";

/**
 * Identity sent as X-User-Id / user_id. Only exists in auth-disabled mode: with
 * Clerk enabled the backend derives identity from the verified session token
 * and rejects a mismatching X-User-Id with 401, so nothing else may send one.
 */
export const DEV_USER_ID: string | undefined = AUTH_DISABLED
  ? (process.env.NEXT_PUBLIC_DEV_USER_ID ?? LOCAL_DEV_USER_ID)
  : undefined;

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
