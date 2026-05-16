import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

// When set, the entire Clerk middleware is bypassed — for self-hosters who
// don't want a Clerk account. Pair with `DISABLE_AUTH=true` on the backend.
const authDisabled = process.env.NEXT_PUBLIC_DISABLE_AUTH === "true";

// Everything not in this list is public (home, about, company pages, marketing).
// Protected routes redirect signed-out visitors to /sign-in.
const isProtectedRoute = createRouteMatcher([
  "/watchlist(.*)",
  "/history(.*)",
  "/session(.*)",
  "/settings(.*)",
  "/companies(.*)",
]);

export default authDisabled
  ? () => NextResponse.next()
  : clerkMiddleware(async (auth, req) => {
      if (isProtectedRoute(req)) {
        await auth.protect();
      }
    });

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
    "/__clerk/(.*)",
  ],
};
