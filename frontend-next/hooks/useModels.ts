"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { EnvKeysResponse, Model } from "@/types";

export function useModels() {
  const [models, setModels] = useState<Model[]>([]);
  const [envKeys, setEnvKeys] = useState<EnvKeysResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.models(), api.envKeys()])
      .then(([modelsRes, envRes]) => {
        setModels(modelsRes.models);
        setEnvKeys(envRes);
      })
      .catch(() => {
        // Backend may not be running; gracefully ignore
      })
      .finally(() => setLoading(false));
  }, []);

  function getDefault(): Model | undefined {
    return models.find((m) => m.default) ?? models[0];
  }

  return { models, envKeys, loading, getDefault };
}
