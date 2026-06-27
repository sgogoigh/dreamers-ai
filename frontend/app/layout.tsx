import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dreamers · AI Trailer Studio",
  description: "Turn a one-line idea into a structured trailer script and a finished trailer video.",
  // app/icon.svg is auto-registered by Next as the favicon — the exact same
  // markup is rendered in the page heading, so tab icon === heading icon.
};

export const viewport: Viewport = {
  themeColor: "#05060d",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
