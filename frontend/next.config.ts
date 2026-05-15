import type { NextConfig } from "next";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS_URL = BACKEND_URL.replace(/^http/, "ws");

// Clerk Frontend API + assets. Dev instances live under *.clerk.accounts.dev;
// production instances under clerk.<your-domain> which resolves via *.clerk.com.
// Telemetry is optional but Clerk's SDK pings it.
const CLERK_CONNECT = [
  "https://*.clerk.accounts.dev",
  "https://*.clerk.com",
  "https://clerk-telemetry.com",
].join(" ");
const CLERK_SCRIPT = "https://*.clerk.accounts.dev https://*.clerk.com https://challenges.cloudflare.com";
const CLERK_FRAME = "https://challenges.cloudflare.com";
const CLERK_IMG = "https://img.clerk.com";

const nextConfig: NextConfig = {
  turbopack: {},
  reactStrictMode: false,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            // Next.js injects inline <script> tags for hydration and page data,
            // so script-src 'self' would break the app. Instead, scope only the
            // directives that are safe to restrict without a nonce setup:
            //   connect-src — prevents XSS data exfiltration to unknown origins
            //   object-src  — blocks plugin-based attacks (Flash, etc.)
            //   frame-ancestors — prevents clickjacking (redundant with X-Frame-Options
            //                     but CSP takes precedence in modern browsers)
            key: "Content-Security-Policy",
            value: [
              `connect-src 'self' ${BACKEND_URL} ${WS_URL} ${CLERK_CONNECT}`,
              `script-src 'self' 'unsafe-inline' 'unsafe-eval' ${CLERK_SCRIPT}`,
              `img-src 'self' data: blob: ${CLERK_IMG}`,
              `frame-src 'self' ${CLERK_FRAME}`,
              "worker-src 'self' blob:",
              "style-src 'self' 'unsafe-inline'",
              "object-src 'none'",
              "frame-ancestors 'none'",
            ].join("; "),
          },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
