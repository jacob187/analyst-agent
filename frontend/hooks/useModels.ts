"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuthReady } from "@/hooks/useAuthReady";
import type { EnvKeysResponse, Model } from "@/types";

export function useModels() {
  const { userId, ready: authReady } = useAuthReady();
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [env, setEnv] = useState<{
    identity: string | null;
    data: EnvKeysResponse | null;
  } | null>(null);

  useEffect(() => {
    api
      .models()
      .then((res) => setModels(res.models))
      .catch(() => {
        // Backend may not be running; gracefully ignore
      })
      .finally(() => setModelsLoading(false));
  }, []);

  // /env-keys answers per caller, so refetch once Clerk settles and whenever
  // the identity changes; a response only counts for the identity it was
  // fetched under.
  useEffect(() => {
    if (!authReady) return;
    let cancelled = false;
    api
      .envKeys()
      .then((data) => { if (!cancelled) setEnv({ identity: userId, data }); })
      .catch(() => { if (!cancelled) setEnv({ identity: userId, data: null }); });
    return () => { cancelled = true; };
  }, [authReady, userId]);

  const envCurrent = env !== null && env.identity === userId;
  const envKeys = envCurrent ? env.data : null;
  const loading = modelsLoading || !envCurrent;

  function getDefault(): Model | undefined {
    return models.find((m) => m.default) ?? models[0];
  }

  return { models, envKeys, loading, getDefault };
}
