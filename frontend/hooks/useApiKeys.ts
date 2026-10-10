"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { api } from "@/lib/api";
import { authDisabled, useAuth } from "@/lib/auth";
import type { ApiKeys, EnvKeysResponse } from "@/types";

const STORAGE_KEY = "analyst_agent_keys";

const DEFAULT_KEYS: ApiKeys = {
  google_api_key: "",
  openai_api_key: "",
  anthropic_api_key: "",
  tavily_api_key: "",
  model_id: "",
};

// Module-level store over localStorage, read via useSyncExternalStore. The
// server snapshot is DEFAULT_KEYS, so hydration matches the SSR markup and
// React swaps in the stored keys right after. In-memory `stored` stays the
// source of truth even if localStorage writes fail.
let stored: ApiKeys | null = null;
const listeners = new Set<() => void>();

function getSnapshot(): ApiKeys {
  if (stored === null) {
    stored = DEFAULT_KEYS;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) stored = { ...DEFAULT_KEYS, ...(JSON.parse(raw) as Partial<ApiKeys>) };
    } catch {
      // Ignore parse errors
    }
  }
  return stored;
}

function getServerSnapshot(): ApiKeys {
  return DEFAULT_KEYS;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function writeKeys(updated: ApiKeys) {
  stored = updated;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch {
    // Ignore storage errors
  }
  listeners.forEach((l) => l());
}

export function useApiKeys() {
  const { isSignedIn, isLoaded: authLoaded } = useAuth();
  const keys = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const [envKeys, setEnvKeys] = useState<EnvKeysResponse | null>(null);
  const [envLoaded, setEnvLoaded] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);

  useEffect(() => {
    // Fetch server-side env key availability
    api.envKeys()
      .then(setEnvKeys)
      .catch(() => {})
      .finally(() => setEnvLoaded(true));

    // A model id saved before a rename would otherwise count as an explicit
    // choice and lock an anonymous visitor out of the free trial.
    api.models()
      .then(({ models }) => {
        const current = getSnapshot();
        if (!current.model_id || models.some((m) => m.id === current.model_id)) return;
        writeKeys({ ...current, model_id: "" });
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
    writeKeys(updated);
  }

  function hasRequiredKeys(): boolean {
    // Operator env provider keys are lent only to signed-in (or self-host)
    // callers — mirror the backend's _env_keys_allowed gate. Counting them for
    // anonymous visitors makes the chat tab render and then fail on WS connect.
    const canUseEnvKeys = !!isSignedIn || authDisabled;

    // Anonymous free trial: the backend lends its key for exactly one light
    // model, capped per-IP. Only counts if the user hasn't picked a different
    // model in Settings — otherwise the chat window would open and then fail
    // on WS connect for a model the trial doesn't cover.
    const usingFreeTrialModel =
      !keys.model_id || keys.model_id === envKeys?.free_trial_model_id;
    const canUseFreeTrial =
      !canUseEnvKeys && !!envKeys?.free_trial_model_id && usingFreeTrialModel;

    return (
      !!keys.google_api_key ||
      !!keys.openai_api_key ||
      !!keys.anthropic_api_key ||
      (canUseEnvKeys &&
        (!!envKeys?.google || !!envKeys?.openai || !!envKeys?.anthropic)) ||
      canUseFreeTrial
    );
  }

  return { keys, setKeys, loaded, envKeys, hasRequiredKeys };
}
