import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Agentic RAG Chat",
  description: "RAG agent chat interfeysi",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="uz">
      <body>{children}</body>
    </html>
  );
}
