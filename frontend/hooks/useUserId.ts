"use client";

import { useEffect } from "react";

import { LOCAL_DEV_USER_ID, authDisabled, useAuth } from "@/lib/auth";

// Identity is sign-in-only. Signed-out visitors have NO user id: they browse
// and use BYOK LLM features freely, but nothing is persisted server-side.
// Signing in (Clerk) is what unlocks DB persistence and cross-device sync.
//
// ACTIVE_KEY mirrors the current Clerk id so the sync getUserId() helper can
// read it from non-component code (lib/api.ts, useWebSocket). It is cleared on
// sign-out so getUserId() never returns a stale id.
const ACTIVE_KEY = "analyst_agent_active_uid";

// Optional stable dev override. When set, every request uses this ID,
// bypassing Clerk entirely. Never set in production.
// `NEXT_PUBLIC_DISABLE_AUTH=true` implies `LOCAL_DEV_USER_ID` automatically.
const DEV_USER_ID = process.env.NEXT_PUBLIC_DEV_USER_ID
  ?? (authDisabled ? LOCAL_DEV_USER_ID : undefined);

let cachedActiveId = "";

function setActiveId(id: string) {
  cachedActiveId = id;
  try {
    if (id) localStorage.setItem(ACTIVE_KEY, id);
    else localStorage.removeItem(ACTIVE_KEY);
  } catch {
    // ignore — localStorage may be unavailable
  }
}

/**
 * Sync read of the current user ID for non-component contexts (fetch helpers
 * in lib/api.ts, the WebSocket hook). Returns the Clerk id when signed in,
 * otherwise "" — signed-out visitors have no identity and send no X-User-Id.
 */
export function getUserId(): string {
  if (DEV_USER_ID) return DEV_USER_ID;
  if (typeof window === "undefined") return "";
  if (cachedActiveId) return cachedActiveId;
  try {
    cachedActiveId = localStorage.getItem(ACTIVE_KEY) ?? "";
  } catch {
    cachedActiveId = "";
  }
  return cachedActiveId;
}

/**
 * React hook that keeps the active user id in sync with Clerk auth state: the
 * Clerk id when signed in, "" when signed out (cache + localStorage cleared so
 * getUserId() never returns a stale id after sign-out). Writes to module cache
 * + localStorage so getUserId() can read it synchronously from non-component code.
 */
export function useUserId() {
  const { userId, isLoaded } = useAuth();

  useEffect(() => {
    if (DEV_USER_ID) {
      setActiveId(DEV_USER_ID);
      return;
    }
    if (!isLoaded) return;
    setActiveId(userId ?? "");
  }, [isLoaded, userId]);

  const effectiveId = DEV_USER_ID ?? (isLoaded ? (userId ?? "") : "");
  return { userId: effectiveId, loaded: isLoaded };
}
