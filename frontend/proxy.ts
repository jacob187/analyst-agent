import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

// When set, the entire Clerk middleware is bypassed — for self-hosters who
// don't want a Clerk account. Pair with `DISABLE_AUTH=true` on the backend.
const authDisabled = process.env.NEXT_PUBLIC_DISABLE_AUTH === "true";

// Progressive auth: most routes are public — signed-out visitors browse, chat
// (BYOK), and read filings anonymously. The exceptions are the routes whose
// only content is per-user persisted data: the backend has no anonymous data
// model for them (require_user_id → 422), so they'd render permanently empty
// with silently no-op actions. Redirect those to sign-in instead. /companies
// and /settings stay public (public data / local BYOK config).
const isProtectedRoute = createRouteMatcher([
  "/watchlist(.*)",
  "/history(.*)",
  "/session(.*)",
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
