"use client";

import { DEV_USER_ID } from "@/lib/constants";
import { useAuth } from "@/lib/auth";

/**
 * Current user id for components: the Clerk id when signed in, "" when signed
 * out, `loaded` once Clerk has resolved. Identity is sign-in-only — signed-out
 * visitors browse and use BYOK LLM features, but nothing is persisted.
 *
 * Display/gating only: requests never send this id. With Clerk enabled the
 * backend derives identity from the session token (lib/api.ts, useWebSocket);
 * the id is sent as X-User-Id / user_id only in auth-disabled mode.
 */
export function useUserId() {
  const { userId, isLoaded } = useAuth();
  const effectiveId = DEV_USER_ID ?? (isLoaded ? (userId ?? "") : "");
  return { userId: effectiveId, loaded: isLoaded };
}
