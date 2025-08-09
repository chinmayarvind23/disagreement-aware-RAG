import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://127.0.0.1:8000/:path*" },
      { source: "/metrics",    destination: "http://127.0.0.1:8000/metrics" },
    ];
  },
};

export default nextConfig;