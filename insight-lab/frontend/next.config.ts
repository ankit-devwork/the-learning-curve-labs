import type { NextConfig } from "next";

/**
 * Vercel + HTTP EC2: browser calls same-origin /api-backend/* (HTTPS).
 * Vercel rewrites proxy to BACKEND_PROXY_URL (server-side, HTTP OK).
 *
 * Local dev: set BACKEND_PROXY_URL=http://localhost:8000 and
 * NEXT_PUBLIC_API_URL=/api-backend — or skip proxy and use
 * NEXT_PUBLIC_API_URL=http://localhost:8000 directly.
 */
const backendProxyUrl = (
  process.env.BACKEND_PROXY_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api-backend/:path*",
        destination: `${backendProxyUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
