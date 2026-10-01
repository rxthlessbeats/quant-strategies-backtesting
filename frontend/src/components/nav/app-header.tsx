"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense } from "react";
import HeaderTickerSearch from "@/components/trading/header-ticker-search";
import { ThemeToggle } from "@/components/theme-toggle";
import { navigations } from "@/config/site";

export default function AppHeader() {
  const pathname = usePathname();
  return (
    <header className="app-header">
      <Link href="/" className="wordmark" aria-label="rookie/trader home">
        <svg viewBox="0 0 36 36" aria-hidden="true" className="brand-mark">
          <rect width="36" height="36" rx="11" fill="currentColor" />
          <path d="M10 25V14h5v11m2-8V10h5v15m2-10h5v10" fill="none" stroke="white" strokeWidth="2.5" />
        </svg>
        <span>rookie<span className="wordmark-slash">/</span>trader</span>
      </Link>
      <nav className="main-nav" aria-label="Main navigation">
        {navigations.map(({ href, name, icon: Icon }) => (
          <Link href={href} key={href} aria-current={pathname === href ? "page" : undefined}>
            <Icon size={15} aria-hidden="true" /><span>{name}</span>
          </Link>
        ))}
      </nav>
      <div className="header-tools">
        <Suspense fallback={<div className="ticker-search" aria-hidden="true"><div className="search-skeleton" /></div>}>
          <HeaderTickerSearch />
        </Suspense>
        <ThemeToggle />
      </div>
    </header>
  );
}
