"use client";

import { Search } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { searchTickers } from "@/lib/api";
import { filterUsEquities } from "@/lib/ticker-search-utils";
import type { TickerSearchItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const DEFAULT_INDICATORS = "sma:5,sma:20";

export default function HeaderTickerSearch() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [symbolInput, setSymbolInput] = useState("");
  const [results, setResults] = useState<TickerSearchItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [shouldSearch, setShouldSearch] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const query = symbolInput.trim();
  const showDropdown =
    dropdownOpen && query.length >= 2 && (searching || results.length > 0);

  useEffect(() => {
    if (!shouldSearch || query.length < 2) {
      setResults([]);
      setSearching(false);
      setDropdownOpen(false);
      return;
    }

    setDropdownOpen(true);
    const timer = setTimeout(() => {
      setSearching(true);
      searchTickers(query)
        .then((response) => setResults(filterUsEquities(response.results)))
        .catch(() => setResults([]))
        .finally(() => setSearching(false));
    }, 450);

    return () => clearTimeout(timer);
  }, [query, shouldSearch]);

  useEffect(() => {
    if (!dropdownOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
        setResults([]);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setDropdownOpen(false);
        setResults([]);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [dropdownOpen]);

  const closeDropdown = () => {
    setDropdownOpen(false);
    setResults([]);
  };

  const navigateToSymbol = (symbol: string) => {
    const normalized = symbol.trim().toUpperCase();
    if (!normalized) {
      return;
    }

    const params = new URLSearchParams({ symbol: normalized });
    const indicators =
      pathname === "/chart"
        ? searchParams.get("indicators")
        : null;
    params.set("indicators", indicators || DEFAULT_INDICATORS);
    router.push(`/chart?${params.toString()}`);
    setSymbolInput("");
    setShouldSearch(false);
    closeDropdown();
  };

  return (
    <div ref={containerRef} className="relative w-full">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          navigateToSymbol(symbolInput);
        }}
      >
        <div className="relative">
          <input
            type="text"
            value={symbolInput}
            onChange={(e) => {
              setSymbolInput(e.target.value.toUpperCase());
              setShouldSearch(true);
            }}
            onFocus={() => {
              if (query.length >= 2) {
                setShouldSearch(true);
                setDropdownOpen(true);
              }
            }}
            className={cn(
              "w-full rounded-md border border-border bg-background",
              "py-2 pl-3 pr-9 text-sm uppercase",
              "placeholder:normal-case placeholder:text-muted-foreground",
            )}
            placeholder="Search US ticker or company"
            autoComplete="off"
          />
          <button
            type="submit"
            aria-label="Search ticker"
            className={cn(
              "absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2",
              "items-center justify-center rounded text-muted-foreground",
              "transition-colors hover:bg-accent hover:text-foreground",
            )}
          >
            <Search className="h-4 w-4" />
          </button>
        </div>
      </form>

      {showDropdown && (
        <div
          className={cn(
            "absolute left-0 top-full z-50 mt-1 w-full",
            "max-h-64 overflow-y-auto rounded-md border border-border",
            "bg-popover py-1 text-popover-foreground shadow-lg",
          )}
        >
          {searching && results.length === 0 && (
            <p className="px-3 py-2 text-xs text-muted-foreground">Searching…</p>
          )}
          {results.map((item) => (
            <button
              key={`${item.symbol}-${item.region ?? ""}`}
              type="button"
              className="flex w-full flex-col px-3 py-2 text-left text-sm hover:bg-accent"
              onClick={() => navigateToSymbol(item.symbol)}
            >
              <span className="font-medium">
                {item.symbol}{" "}
                <span className="font-normal text-muted-foreground">
                  {item.name}
                </span>
              </span>
              {item.region && (
                <span className="text-xs text-muted-foreground">
                  {item.region}
                </span>
              )}
            </button>
          ))}
          {!searching && results.length === 0 && (
            <p className="px-3 py-2 text-xs text-muted-foreground">
              No US stocks found
            </p>
          )}
        </div>
      )}
    </div>
  );
}
