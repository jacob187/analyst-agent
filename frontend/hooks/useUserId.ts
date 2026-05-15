"use client";

import { useAuth } from "@clerk/nextjs";
import { useEffect } from "react";

// Cached Clerk user ID under a separate key. The old `analyst_agent_user_id`
// UUID key is intentionally left untouched — Phase 1.5 migrates it.
const CACHE_KEY = "clerk_user_id";

// Optional stable dev override. When set, every request uses this ID,
// bypassing Clerk entirely. Never set in production.
const DEV_USER_ID = process.env.NEXT_PUBLIC_DEV_USER_ID;

let cachedUserId = "";

function setCachedUserId(id: string) {
  cachedUserId = id;
  try {
    localStorage.setItem(CACHE_KEY, id);
  } catch {
    // ignore — localStorage may be unavailable
  }
}

/**
 * Sync read of the current user ID for non-component contexts (e.g. fetch
 * helpers in lib/api.ts). Returns "" if Clerk hasn't loaded yet on this
 * client. Prefer useUserId() inside components.
 */
export function getUserId(): string {
  if (DEV_USER_ID) return DEV_USER_ID;
  if (cachedUserId) return cachedUserId;
  try {
    const stored = localStorage.getItem(CACHE_KEY);
    if (stored) {
      cachedUserId = stored;
      return stored;
    }
  } catch {
    // ignore
  }
  return "";
}

/**
 * React hook returning the current Clerk user ID and a loaded flag.
 * Writes the ID to module cache + localStorage so getUserId() can read it
 * synchronously from non-component code.
 */
export function useUserId() {
  const { userId, isLoaded } = useAuth();

  useEffect(() => {
    if (DEV_USER_ID) {
      setCachedUserId(DEV_USER_ID);
      return;
    }
    if (isLoaded && userId) {
      setCachedUserId(userId);
    }
  }, [isLoaded, userId]);

  const effectiveId = DEV_USER_ID ?? userId ?? "";
  return { userId: effectiveId, loaded: isLoaded };
}
