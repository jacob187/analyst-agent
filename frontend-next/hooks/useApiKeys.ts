"use client";

import { useEffect, useState } from "react";
import type { ApiKeys } from "@/types";

const STORAGE_KEY = "analyst_agent_keys";

const DEFAULT_KEYS: ApiKeys = {
  google_api_key: "",
  openai_api_key: "",
  anthropic_api_key: "",
  sec_header: "",
  tavily_api_key: "",
  model_id: "",
};

export function useApiKeys() {
  const [keys, setKeysState] = useState<ApiKeys>(DEFAULT_KEYS);
  const [loaded, setLoaded] = useState(false);

  // Load from localStorage on mount (SSR-safe: only runs client-side)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<ApiKeys>;
        setKeysState({ ...DEFAULT_KEYS, ...parsed });
      }
    } catch {
      // Ignore parse errors
    }
    setLoaded(true);
  }, []);

  function setKeys(updated: ApiKeys) {
    setKeysState(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // Ignore storage errors
    }
  }

  function hasRequiredKeys(): boolean {
    const hasProviderKey =
      !!keys.google_api_key ||
      !!keys.openai_api_key ||
      !!keys.anthropic_api_key;
    return hasProviderKey && !!keys.sec_header;
  }

  return { keys, setKeys, loaded, hasRequiredKeys };
}
