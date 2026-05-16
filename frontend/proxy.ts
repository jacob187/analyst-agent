import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Everything not in this list is public (home, about, company pages, marketing).
// Protected routes redirect signed-out visitors to /sign-in.
const isProtectedRoute = createRouteMatcher([
  "/watchlist(.*)",
  "/history(.*)",
  "/session(.*)",
  "/settings(.*)",
  "/companies(.*)",
]);

export default clerkMiddleware(async (auth, req) => {
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
