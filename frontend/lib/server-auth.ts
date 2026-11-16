import { auth } from "@clerk/nextjs/server";

import { AUTH_DISABLED, DEV_USER_ID } from "./constants";

/**
 * Identity headers for API calls made from server components. The browser-side
 * helpers in lib/api.ts read `window.Clerk`, which doesn't exist here, so
 * server pages must forward the request's Clerk session token explicitly.
 *
 * Returns `{}` when signed out (the backend then treats the call as anonymous).
 * Server-only: import from server components / route handlers, never from
 * client code.
 */
export async function serverAuthHeaders(): Promise<Record<string, string>> {
  if (AUTH_DISABLED) {
    return DEV_USER_ID ? { "X-User-Id": DEV_USER_ID } : {};
  }
  try {
    const { getToken, userId } = await auth();
    const token = await getToken();
    if (!token) return {};
    // The id must equal the token's subject; older backends read only the id.
    return userId
      ? { "X-Clerk-Session-Token": token, "X-User-Id": userId }
      : { "X-Clerk-Session-Token": token };
  } catch (err) {
    // Clerk unconfigured or middleware not applied: fall back to anonymous.
    console.error("serverAuthHeaders: could not read Clerk session", err);
    return {};
  }
}
