"use client";

import { ArrowUpRight, Search, SlidersHorizontal, ChevronDown, Copy, Check, RotateCcw } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import type { IndicatorCatalogItem } from "@/lib/types";

function queryFor(item: IndicatorCatalogItem) {
  const entries = Object.entries(item.params);
  if (!entries.length) return item.id;
  return entries.length === 1 && entries[0][0] === "period" ? `${item.id}:${entries[0][1]}` : `${item.id}:${entries.map(([key, value]) => `${key}=${value}`).join(";")}`;
}

export default function IndicatorLibrary({ items, error }: { items: IndicatorCatalogItem[]; error: string | null }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All signals");
  const [copied, setCopied] = useState<string | null>(null);
  const [copyError, setCopyError] = useState<string | null>(null);
  const categories = ["All signals", ...Array.from(new Set(items.map(item => item.category)))];
  const filtered = useMemo(() => items.filter(item => (category === "All signals" || item.category === category) && `${item.id} ${item.description} ${item.category}`.toLowerCase().includes(query.trim().toLowerCase())), [items, category, query]);

  return (
    <div className="library-page">
      <div className="library-heading"><div><p className="context-label">The indicator library</p><h1>Different signals.<br />A deeper perspective.</h1></div><p>A toolkit for reading the market.<br />Find a signal, tune its parameters,<br />and see what it adds to your view.</p></div>
      <div className="library-controls">
        <div className="category-tabs" aria-label="Filter indicators by category">{categories.map(value => <button key={value} type="button" aria-pressed={category === value} onClick={() => setCategory(value)}>{value}</button>)}</div>
        <div className="library-search"><Search size={16} aria-hidden="true" /><input type="search" aria-label="Search indicators" value={query} onChange={event => setQuery(event.target.value)} placeholder="Find an indicator" /></div>
      </div>
      <div className="library-count" role="status">{error ? "Catalog unavailable" : `${filtered.length} ${filtered.length === 1 ? "indicator" : "indicators"}${category !== "All signals" ? ` in ${category}` : " to explore"}`}</div>
      {error ? <div className="catalog-empty"><SlidersHorizontal size={28} /><h2>The toolkit is taking a moment.</h2><p>{error}</p><a href="/indicators">Reload catalog <RotateCcw size={15} /></a></div> : filtered.length === 0 ? <div className="catalog-empty"><Search size={28} /><h2>No signals match that search.</h2><p>Try another name or explore a different category.</p><button type="button" onClick={() => { setQuery(""); setCategory("All signals"); }}>Show all indicators</button></div> : <div className="indicator-list">{filtered.map(item => {
        const example = queryFor(item);
        return <details key={item.id} className="indicator-item"><summary><span className={`signal-glyph signal-${item.category.toLowerCase()}`} aria-hidden="true"><svg viewBox="0 0 56 40"><path d={item.category.toLowerCase() === "volume" ? "M7 32V23M17 32V14M27 32V19M37 32V8M47 32V16" : item.category.toLowerCase() === "momentum" ? "M4 24C12 3 17 3 23 20S36 38 42 17S49 8 53 15" : item.category.toLowerCase() === "volatility" ? "M4 24L11 15L18 23L25 6L32 31L39 12L46 24L53 19" : "M4 31L12 26L20 28L28 18L36 21L44 12L52 7"} /></svg></span><div className="indicator-name"><h2>{item.id.toUpperCase()}</h2><span>{item.category}</span></div><p>{item.description}</p><span className="indicator-action">Explore <ChevronDown size={18} /></span></summary>
          <div className="indicator-detail"><div><h3>Default parameters</h3><dl>{Object.entries(item.params).length ? Object.entries(item.params).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>) : <div><dt>No parameters required</dt></div>}</dl><p>Parameters can be adjusted in the workspace.</p></div><div><h3>Query example</h3><div className="query-example"><code>{example}</code><button type="button" aria-label={`Copy ${item.id} query`} onClick={async () => { try { await navigator.clipboard.writeText(example); setCopied(item.id); setCopyError(null); } catch { setCopyError(item.id); } }}>{copied === item.id ? <Check size={15} /> : <Copy size={15} />}</button></div>{copied === item.id && <p role="status">Query copied.</p>}{copyError === item.id && <p role="status">Select the query above to copy it.</p>}<Link href={`/chart?${new URLSearchParams({ symbol: "NVDA", indicators: example })}`}>Explore in workspace <ArrowUpRight size={16} /></Link></div></div>
        </details>;
      })}</div>}
      <div className="library-footnote"><SlidersHorizontal size={16} /><p>Combine multiple indicators in the workspace. Your saved preset keeps your view ready for next time.</p><Link href="/chart">Open workspace <ArrowUpRight size={15} /></Link></div>
    </div>
  );
}
