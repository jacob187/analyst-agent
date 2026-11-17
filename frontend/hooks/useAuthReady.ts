"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";

/** How long identity-dependent work waits for Clerk before proceeding anonymously. */
export const AUTH_WAIT_MS = 4000;

/**
 * Clerk's auth state, plus a bounded wait. `ready` flips true once Clerk has
 * loaded, or after `timeoutMs` if it never does (blocked script, network
 * failure), so callers can proceed anonymously instead of hanging. `userId`
 * keeps updating afterwards, so a late-loading Clerk still triggers the
 * identity change callers key their effects on.
 */
export function useAuthReady(timeoutMs = AUTH_WAIT_MS) {
  const { isLoaded, isSignedIn, userId } = useAuth();
  const [timedOut, setTimedOut] = useState(false);

  useEffect(() => {
    if (isLoaded) return;
    const t = setTimeout(() => setTimedOut(true), timeoutMs);
    return () => clearTimeout(t);
  }, [isLoaded, timeoutMs]);

  return {
    isLoaded,
    isSignedIn,
    userId: userId ?? null,
    ready: isLoaded || timedOut,
    /** Clerk did not load within the wait; the caller is acting anonymously. */
    authFailed: timedOut && !isLoaded,
  };
}
