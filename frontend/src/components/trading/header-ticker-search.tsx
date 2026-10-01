"use client";

import { Search, Loader2 } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import { searchTickers } from "@/lib/api";
import { filterUsEquities } from "@/lib/ticker-search-utils";
import type { TickerSearchItem } from "@/lib/types";

export default function HeaderTickerSearch() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [input, setInput] = useState("");
  const [results, setResults] = useState<TickerSearchItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState(false);
  const [active, setActive] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listId = useId();
  const query = input.trim();
  const showDropdown = open && query.length >= 2;

  useEffect(() => {
    let cancelled = false;
    setResults([]); setActive(-1); setError(false);
    if (!open || query.length < 2) { setSearching(false); return; }
    setSearching(true);
    const timer = window.setTimeout(() => {
      searchTickers(query).then(response => { if (!cancelled) setResults(filterUsEquities(response.results)); })
        .catch(() => { if (!cancelled) setError(true); })
        .finally(() => { if (!cancelled) setSearching(false); });
    }, 350);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [query, open]);

  useEffect(() => {
    const pointer = (event: PointerEvent) => { if (!containerRef.current?.contains(event.target as Node)) setOpen(false); };
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); inputRef.current?.focus(); }
    };
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("keydown", keyboard);
    return () => { document.removeEventListener("pointerdown", pointer); document.removeEventListener("keydown", keyboard); };
  }, []);

  const navigate = (symbol: string) => {
    const normalized = symbol.trim().toUpperCase();
    if (!normalized) return;
    const params = new URLSearchParams({ symbol: normalized, indicators: (pathname === "/chart" ? searchParams.get("indicators") : null) ?? "sma:5,sma:20" });
    router.push(`/chart?${params}`);
    setInput(""); setOpen(false); inputRef.current?.blur();
  };

  return (
    <div ref={containerRef} className="ticker-search">
      <form onSubmit={e => { e.preventDefault(); navigate(active >= 0 && results[active] ? results[active].symbol : input); }}>
        <Search size={16} className="search-icon" aria-hidden="true" />
        <input ref={inputRef} type="text" value={input} role="combobox" aria-label="Search US ticker or company" aria-autocomplete="list" aria-expanded={showDropdown} aria-controls={showDropdown ? listId : undefined} aria-activedescendant={active >= 0 && showDropdown ? `${listId}-${active}` : undefined}
          onChange={e => { setInput(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)}
          onKeyDown={e => {
            if (e.key === "ArrowDown" && results.length) { e.preventDefault(); setActive(value => (value + 1) % results.length); }
            if (e.key === "ArrowUp" && results.length) { e.preventDefault(); setActive(value => (value - 1 + results.length) % results.length); }
          }} placeholder="Search a company" autoComplete="off" />
        {searching ? <Loader2 size={15} className="search-shortcut animate-spin" aria-hidden="true" /> : <span className="search-shortcut" aria-hidden="true">⌘ K</span>}
      </form>
      {showDropdown && <div className="search-dropdown">
        <div role="listbox" id={listId} aria-label="Ticker results">
          {results.map((item, index) => <button key={item.symbol} type="button" role="option" id={`${listId}-${index}`} aria-selected={active === index} tabIndex={-1} onPointerDown={e => e.preventDefault()} onClick={() => navigate(item.symbol)}><strong>{item.symbol}</strong><span>{item.name}</span></button>)}
        </div>
        {!results.length && <p role="status">{searching ? "Searching US equities…" : error ? "Search unavailable. Enter a ticker to open its workspace." : "No matching US stocks. Enter a ticker to open it directly."}</p>}
        <div className="search-help">↑ ↓ to browse <span>Enter to open</span></div>
      </div>}
    </div>
  );
}
