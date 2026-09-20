import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Inbox",
  description: "A minimal interface for talking through tasks, ideas, and reminders.",
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
