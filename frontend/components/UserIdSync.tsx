"use client";

import { useUserId } from "@/hooks/useUserId";

/**
 * Mounts the useUserId() sync globally (from the root layout) so getUserId()'s
 * module cache + localStorage track Clerk auth on EVERY route. Previously the
 * sync ran only on pages that happened to call useUserId() (/companies,
 * /history), so a signed-in user landing directly on /company/[ticker] (the
 * primary flow) had getUserId() === "" and was silently treated as anonymous —
 * no chat persistence, no session resume, denied the operator key. Renders nothing.
 */
export function UserIdSync() {
  useUserId();
  return null;
}
