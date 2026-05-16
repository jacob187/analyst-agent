"use client";

/**
 * Auth shim that lets the app run with OR without Clerk.
 *
 * When `NEXT_PUBLIC_DISABLE_AUTH=true`, all Clerk components and hooks are
 * replaced with no-op fallbacks so the app works for self-hosters who don't
 * want to create a Clerk account. Pair with `DISABLE_AUTH=true` on the
 * backend so JWT verification is also skipped.
 *
 * Everything else in the frontend imports auth primitives from this module
 * instead of `@clerk/nextjs` directly, so swapping the auth provider only
 * touches this file.
 */

import {
  Show as ClerkShow,
  SignInButton as ClerkSignInButton,
  UserButton as ClerkUserButton,
  useAuth as clerkUseAuth,
} from "@clerk/nextjs";
import type { ReactNode } from "react";

export const authDisabled = process.env.NEXT_PUBLIC_DISABLE_AUTH === "true";

/** Stable user_id used when auth is disabled — matches the backend's USER_ID_RE. */
export const LOCAL_DEV_USER_ID = "user_localdev";

interface UseAuthShape {
  isLoaded: boolean;
  isSignedIn: boolean | undefined;
  userId: string | null | undefined;
}

function noAuthUseAuth(): UseAuthShape {
  return { isLoaded: true, isSignedIn: true, userId: LOCAL_DEV_USER_ID };
}

export const useAuth: () => UseAuthShape = authDisabled
  ? noAuthUseAuth
  : (clerkUseAuth as () => UseAuthShape);

interface ShowProps {
  when: "signed-in" | "signed-out";
  children: ReactNode;
}

export function Show({ when, children }: ShowProps) {
  if (authDisabled) {
    // Treat the local-dev user as always-signed-in.
    return when === "signed-in" ? <>{children}</> : null;
  }
  return <ClerkShow when={when}>{children}</ClerkShow>;
}

interface SignInButtonProps {
  mode?: "modal" | "redirect";
  children?: ReactNode;
}

export function SignInButton({ mode, children }: SignInButtonProps) {
  if (authDisabled) return null;
  return <ClerkSignInButton mode={mode}>{children}</ClerkSignInButton>;
}

export function UserButton(props: React.ComponentProps<typeof ClerkUserButton>) {
  if (authDisabled) return null;
  return <ClerkUserButton {...props} />;
}
