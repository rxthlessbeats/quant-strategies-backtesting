"use client";

import Image from "next/image";
import Link from "next/link";
import { Suspense } from "react";
import Container from "@/components/container";
import HeaderTickerSearch from "@/components/trading/header-ticker-search";
import { ThemeToggle } from "@/components/theme-toggle";
import { siteConfig } from "@/config/site";

function HeaderSearchFallback() {
  return (
    <div className="h-9 w-full animate-pulse rounded-md bg-muted" />
  );
}

export default function AppHeader() {
  return (
    <header className="sticky top-0 z-40 h-16 shrink-0 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <Container className="flex h-full items-center gap-3 sm:gap-4">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2 text-foreground"
        >
          <Image
            src="/RookieTraderLogo.png"
            alt={siteConfig.title}
            width={36}
            height={36}
            className="h-9 w-9 object-contain"
            priority
          />
          <span className="hidden text-lg font-semibold tracking-tight sm:inline">
            {siteConfig.title}
          </span>
        </Link>

        <div className="min-w-0 flex-1 max-w-md">
          <Suspense fallback={<HeaderSearchFallback />}>
            <HeaderTickerSearch />
          </Suspense>
        </div>

        <div className="ml-auto shrink-0">
          <ThemeToggle />
        </div>
      </Container>
    </header>
  );
}
