"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { api } from "@/lib/api";
import { useAuthReady } from "@/hooks/useAuthReady";
import { AUTH_DISABLED } from "@/lib/constants";
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

const PROVIDER_ENV_FLAG = {
  google_genai: "google",
  openai: "openai",
  anthropic: "anthropic",
} as const;

/**
 * Ids of the models the server's own keys pay for, for this caller. The
 * server's `operator_models` is the source of truth — it already folds in key
 * presence, the lending rule and who is asking. A backend that predates the
 * field only reports per-provider flags, so those stand in for it.
 */
export function operatorModelIds(
  envKeys: EnvKeysResponse,
  models: Model[]
): Set<string> {
  if (envKeys.operator_models) return new Set(envKeys.operator_models);
  return new Set(
    models.filter((m) => envKeys[PROVIDER_ENV_FLAG[m.provider]]).map((m) => m.id)
  );
}

/** True when the user has their own key for `model`'s provider, or the server's keys pay for it. */
export function modelHasKey(
  model: Model,
  keys: ApiKeys,
  operatorModels: Set<string>
): boolean {
  return !!keys[PROVIDER_KEY_FIELD[model.provider]] || operatorModels.has(model.id);
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
  const { isSignedIn, userId, ready: authReady } = useAuthReady();
  const keys = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const [env, setEnv] = useState<{
    identity: string | null;
    data: EnvKeysResponse | null;
  } | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);

  // /env-keys answers per caller (the server's keys are only lent to signed-in
  // or self-host callers), so it is fetched once Clerk has settled and again
  // whenever the identity changes. A response is only used for the identity it
  // was fetched under, so a stale anonymous answer never reaches a signed-in user.
  useEffect(() => {
    if (!authReady) return;
    let cancelled = false;
    api.envKeys()
      .then((data) => { if (!cancelled) setEnv({ identity: userId, data }); })
      .catch(() => { if (!cancelled) setEnv({ identity: userId, data: null }); });
    return () => { cancelled = true; };
  }, [authReady, userId]);

  const envCurrent = env !== null && env.identity === userId;
  const envKeys = envCurrent ? env.data : null;

  useEffect(() => {
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

  // Gate consumers until the stale-model check has resolved and /env-keys has
  // answered for the current identity (which implies Clerk has settled), so
  // hasRequiredKeys() never runs on an anonymous answer for a signed-in user
  // (flashing the "sign in / add keys" panel) or against a model id about to
  // be cleared. If Clerk never loads, the wait times out and the anonymous
  // answer is used.
  const loaded = envCurrent && modelsLoaded;

  // Mirrors the backend's resolve_model: an explicitly saved model runs on its
  // own key or not at all; with no saved choice the backend picks whichever
  // model a key can run, so one runnable model is enough. Which models the
  // server's keys pay for comes from the server (operatorModelIds), not from a
  // client-side copy of its lending rule.
  const operatorModels = envKeys ? operatorModelIds(envKeys, models) : new Set<string>();
  const explicitModel = models.find((m) => m.id === keys.model_id);
  const ownKeyModel =
    !!isSignedIn && !AUTH_DISABLED && envKeys && explicitModel &&
    !modelHasKey(explicitModel, keys, operatorModels)
      ? explicitModel
      : null;

  function setKeys(updated: ApiKeys) {
    writeKeys(updated);
  }

  function hasRequiredKeys(): boolean {
    // Some key can run the model the backend will pick. The server reports
    // operator coverage only to callers allowed to spend its keys (signed-in or
    // self-host), so anonymous visitors never count it — otherwise the chat tab
    // would render and then fail on WS connect.
    const hasRunnableKey = explicitModel
      ? modelHasKey(explicitModel, keys, operatorModels)
      : models.length > 0
        ? models.some((m) => modelHasKey(m, keys, operatorModels))
        : // Model list unavailable: any own key is the best signal left.
          Object.values(PROVIDER_KEY_FIELD).some((field) => !!keys[field]);

    // Anonymous free trial: the backend lends its key for exactly one light
    // model, capped per-IP. Only counts if the user hasn't picked a different
    // model in Settings — otherwise the chat window would open and then fail
    // on WS connect for a model the trial doesn't cover.
    const usingFreeTrialModel =
      !keys.model_id || keys.model_id === envKeys?.free_trial_model_id;
    const canUseFreeTrial =
      !isSignedIn &&
      !AUTH_DISABLED &&
      !!envKeys?.free_trial_model_id &&
      usingFreeTrialModel;

    return hasRunnableKey || canUseFreeTrial;
  }

  return { keys, setKeys, loaded, envKeys, hasRequiredKeys, ownKeyModel };
}
