"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { api } from "@/lib/api";
import { authDisabled, useAuth } from "@/lib/auth";
import type { ApiKeys, EnvKeysResponse, Model } from "@/types";

const STORAGE_KEY = "analyst_agent_keys";

const DEFAULT_KEYS: ApiKeys = {
  google_api_key: "",
  openai_api_key: "",
  anthropic_api_key: "",
  tavily_api_key: "",
  model_id: "",
};

export const PROVIDER_KEY_FIELD: Record<Model["provider"], keyof ApiKeys> = {
  google_genai: "google_api_key",
  openai: "openai_api_key",
  anthropic: "anthropic_api_key",
};

/**
 * True when the server won't lend its key for `model` and the user hasn't
 * supplied their own key for that model's provider. Callers add the identity
 * conditions (signed in, auth enabled) — self-host and anonymous callers are
 * not subject to the operator-lendable rule. An absent `operator_lendable`
 * (older backend) counts as lendable so we never block on missing data.
 */
export function modelNeedsOwnKey(model: Model | undefined, keys: ApiKeys): boolean {
  if (!model || model.operator_lendable !== false) return false;
  return !keys[PROVIDER_KEY_FIELD[model.provider]];
}

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
  const [models, setModels] = useState<Model[]>([]);
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
        setModels(models);
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

  // The model the backend will run: the saved choice, else the registry
  // default. Signed-in callers on a hosted deploy need their own key when it
  // isn't operator-lendable; self-host (auth disabled) runs everything on the
  // server's keys.
  const selectedModel =
    models.find((m) => m.id === keys.model_id) ??
    models.find((m) => m.default) ??
    models[0];
  const ownKeyModel =
    !!isSignedIn && !authDisabled && modelNeedsOwnKey(selectedModel, keys)
      ? selectedModel
      : null;

  function setKeys(updated: ApiKeys) {
    writeKeys(updated);
  }

  function hasRequiredKeys(): boolean {
    // Operator env provider keys are lent only to signed-in (or self-host)
    // callers — mirror the backend's _env_keys_allowed gate. Counting them for
    // anonymous visitors makes the chat tab render and then fail on WS connect.
    const canUseEnvKeys = !!isSignedIn || authDisabled;

    // A non-lendable model never runs on the operator's key, whatever env keys
    // exist — chat would connect and then fail. ownKeyModel is set only when
    // the user has no own key for that model's provider.
    if (ownKeyModel) return false;

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

  return { keys, setKeys, loaded, envKeys, hasRequiredKeys, ownKeyModel };
}
