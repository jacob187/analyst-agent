"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ApiKeys, EnvKeysResponse } from "@/types";

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
  const [envKeys, setEnvKeys] = useState<EnvKeysResponse | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    // Load local keys from localStorage
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<ApiKeys>;
        setKeysState({ ...DEFAULT_KEYS, ...parsed });
      }
    } catch {
      // Ignore parse errors
    }

    // Fetch server-side env key availability
    api.envKeys()
      .then(setEnvKeys)
      .catch(() => {})
      .finally(() => setLoaded(true));
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
    // Keys can come from localStorage (user-entered) OR server .env
    const hasProviderKey =
      !!keys.google_api_key ||
      !!keys.openai_api_key ||
      !!keys.anthropic_api_key ||
      !!envKeys?.google ||
      !!envKeys?.openai ||
      !!envKeys?.anthropic;
    const hasSecHeader = !!keys.sec_header || !!envKeys?.sec_header;
    return hasProviderKey && hasSecHeader;
  }

  return { keys, setKeys, loaded, envKeys, hasRequiredKeys };
}
