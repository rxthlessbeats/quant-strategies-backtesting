import type { Metadata } from "next";
import { Gabarito } from "next/font/google";
import AppHeader from "@/components/nav/app-header";
import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";
import "@/style/globals.css";
import { Providers } from "./providers";

const gabarito = Gabarito({ subsets: ["latin"], variable: "--font-gabarito" });

export const metadata: Metadata = {
  title: siteConfig.title,
  description: siteConfig.description,
  icons: {
    icon: "/rookie-mark.svg",
    apple: "/RookieTraderLogo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn("bg-background font-sans", gabarito.variable)}>
        <Providers>
          <div className="app-shell">
            <a href="#main-content" className="skip-link">Skip to content</a>
            <AppHeader />
            {children}
            <footer className="app-footer">
              <span>Rookie Trader <span className="footer-divider">/</span> A little perspective goes a long way.</span>
              <span>Daily market data <span className="footer-divider">/</span> Built for the curious.</span>
            </footer>
          </div>
        </Providers>
      </body>
    </html>
  );
}
