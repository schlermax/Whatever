import type { NextConfig } from "next";

const isProd = process.env.NEXT_PUBLIC_ENVIRONMENT === "production";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: isProd ? "export" : undefined,
  basePath: isProd ? "/Next.js-Lesson-Plans" : "",
  assetPrefix: isProd ? "/Next.js-Lesson-Plans/" : "",
  images: {
    unoptimized: isProd,
  },
};

export default nextConfig;