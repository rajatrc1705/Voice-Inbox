import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Underwriting Voice Agent",
  description: "A minimal interface for testing an underwriting voice agent.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
