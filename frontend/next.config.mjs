/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The backend is reached directly via NEXT_PUBLIC_API_BASE (default
  // http://localhost:8000). It already sends permissive CORS headers, so
  // cross-origin fetch + EventSource (SSE) work without a proxy.
};

export default nextConfig;
