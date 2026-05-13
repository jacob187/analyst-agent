"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "analyst_agent_user_id";

// Optional stable dev user ID. When set (e.g. in .env.local), every
// request uses this ID instead of the per-browser localStorage UUID —
// so clearing storage or switching browsers doesn't fragment local data.
// Never set this in production builds.
const DEV_USER_ID = process.env.NEXT_PUBLIC_DEV_USER_ID;

/**
 * Get or create a stable anonymous user ID from localStorage.
 *
 * This is a plain function (not a hook) so it can be called from
 * non-component contexts like api.ts fetch helpers. It reads
 * synchronously from localStorage — safe in any client-side code,
 * but must not be called during SSR.
 */
export function getUserId(): string {
  if (DEV_USER_ID) return DEV_USER_ID;
  try {
    const existing = localStorage.getItem(STORAGE_KEY);
    if (existing) return existing;

    const id = crypto.randomUUID();
    localStorage.setItem(STORAGE_KEY, id);
    return id;
  } catch {
    // Fallback for environments where localStorage is unavailable
    // (e.g., SSR or restrictive privacy settings). Generate a
    // per-session ID that won't persist across page loads.
    return crypto.randomUUID();
  }
}

/**
 * React hook wrapper around getUserId(). Returns the user ID and
 * a loaded flag (false during SSR, true after hydration).
 */
export function useUserId() {
  const [userId, setUserId] = useState("");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setUserId(getUserId());
    setLoaded(true);
  }, []);

  return { userId, loaded };
}
