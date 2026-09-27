"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { authDisabled, useAuth } from "@/lib/auth";
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
  const { isSignedIn, isLoaded: authLoaded } = useAuth();
  const [keys, setKeysState] = useState<ApiKeys>(DEFAULT_KEYS);
  const [envKeys, setEnvKeys] = useState<EnvKeysResponse | null>(null);
  const [envLoaded, setEnvLoaded] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);

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
      .finally(() => setEnvLoaded(true));

    // A model id saved before a rename would otherwise count as an explicit
    // choice and lock an anonymous visitor out of the free trial.
    api.models()
      .then(({ models }) => {
        setKeysState((prev) => {
          if (!prev.model_id || models.some((m) => m.id === prev.model_id)) return prev;
          const next = { ...prev, model_id: "" };
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch {
            // Ignore storage errors
          }
          return next;
        });
      })
      .catch(() => {})
      .finally(() => setModelsLoaded(true));
  }, []);

  // Gate consumers until the env-key probe, stale-model check, and Clerk auth
  // have all resolved, so hasRequiredKeys() never runs while isSignedIn ===
  // undefined (flashing the "sign in / add keys" panel at a signed-in user
  // relying on the env key) or against a model id about to be cleared.
  const loaded = envLoaded && modelsLoaded && authLoaded;

  function setKeys(updated: ApiKeys) {
    setKeysState(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // Ignore storage errors
    }
  }

  function hasRequiredKeys(): boolean {
    // Operator env provider keys are lent only to signed-in (or self-host)
    // callers — mirror the backend's _env_keys_allowed gate. Counting them for
    // anonymous visitors makes the chat tab render and then fail on WS connect.
    // SEC_HEADER is a global server identity, not a per-user key, so it's
    // available to everyone.
    const canUseEnvKeys = !!isSignedIn || authDisabled;

    // Anonymous free trial: the backend lends its key for exactly one light
    // model, capped per-IP. Only counts if the user hasn't picked a different
    // model in Settings — otherwise the chat window would open and then fail
    // on WS connect for a model the trial doesn't cover.
    const usingFreeTrialModel =
      !keys.model_id || keys.model_id === envKeys?.free_trial_model_id;
    const canUseFreeTrial =
      !canUseEnvKeys && !!envKeys?.free_trial_model_id && usingFreeTrialModel;

    const hasProviderKey =
      !!keys.google_api_key ||
      !!keys.openai_api_key ||
      !!keys.anthropic_api_key ||
      (canUseEnvKeys &&
        (!!envKeys?.google || !!envKeys?.openai || !!envKeys?.anthropic)) ||
      canUseFreeTrial;
    const hasSecHeader = !!keys.sec_header || !!envKeys?.sec_header;
    return hasProviderKey && hasSecHeader;
  }

  return { keys, setKeys, loaded, envKeys, hasRequiredKeys };
}
