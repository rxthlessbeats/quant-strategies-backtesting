"use client";

import { Loader2, Plus, Search } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { fetchIndicatorCatalog } from "@/lib/api";
import { CHART_VIEW_OPTIONS, type ChartView } from "@/lib/chart-view";
import type { ChartMeta } from "@/lib/types";
import type { IndicatorCatalogItem } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ChartToolbarProps {
  symbol: string;
  onSymbolChange: (symbol: string) => void;
  onSymbolSubmit: () => void;
  selectedView: ChartView;
  onViewChange: (view: ChartView) => void;
  loading: boolean;
  meta?: ChartMeta;
  onAddIndicatorPick: (
    id: string,
    defaultPeriod: number,
    anchor: HTMLElement,
  ) => void;
}

export default function ChartToolbar({
  symbol,
  onSymbolChange,
  onSymbolSubmit,
  selectedView,
  onViewChange,
  loading,
  meta,
  onAddIndicatorPick,
}: ChartToolbarProps) {
  const [catalog, setCatalog] = useState<IndicatorCatalogItem[]>([]);
  const addButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    fetchIndicatorCatalog()
      .then(setCatalog)
      .catch(() => setCatalog([]));
  }, []);

  return (
    <form
      className="flex flex-wrap items-center gap-2 px-3 py-2"
      onSubmit={(e) => {
        e.preventDefault();
        onSymbolSubmit();
      }}
    >
      <div className="relative">
        <input
          type="text"
          value={symbol}
          onChange={(e) => onSymbolChange(e.target.value.toUpperCase())}
          className="w-28 rounded border border-white/15 bg-black/40 py-1 pl-2 pr-8 text-sm uppercase text-slate-100"
          placeholder="AAPL"
        />
        <button
          type="submit"
          aria-label="Search ticker"
          className={cn(
            "absolute right-1 top-1/2 -translate-y-1/2 rounded p-1",
            "text-slate-400 hover:bg-white/10 hover:text-slate-100",
          )}
        >
          <Search className="h-3.5 w-3.5" />
        </button>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            ref={addButtonRef}
            type="button"
            variant="ghost"
            size="sm"
            className={cn(
              "h-8 gap-1.5 px-2 text-xs text-slate-300",
              "hover:bg-white/10 hover:text-slate-100",
            )}
          >
            <Plus className="h-3.5 w-3.5" />
            Add indicator
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[140px]">
          {catalog.length === 0 ? (
            <DropdownMenuItem disabled>No indicators</DropdownMenuItem>
          ) : (
            catalog.map((item) => {
              const period =
                typeof item.params.period === "number"
                  ? item.params.period
                  : 20;
              return (
                <DropdownMenuItem
                  key={item.id}
                  onSelect={() => {
                    const anchor = addButtonRef.current;
                    if (!anchor) return;
                    window.setTimeout(() => {
                      onAddIndicatorPick(item.id, period, anchor);
                    }, 0);
                  }}
                >
                  <span className="lowercase">{item.id}</span>
                </DropdownMenuItem>
              );
            })
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <div className="flex items-center gap-1 rounded-md border border-white/10 bg-black/20 p-1">
        {CHART_VIEW_OPTIONS.map((view) => (
          <Button
            key={view}
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => onViewChange(view)}
            className={cn(
              "h-7 px-2 text-[11px] text-slate-400",
              selectedView === view && "bg-white/10 text-slate-100",
            )}
          >
            {view}
          </Button>
        ))}
      </div>
      {loading && (
        <Loader2 className="ml-auto h-4 w-4 animate-spin text-slate-400" />
      )}
      {meta && !loading && (
        <span className="ml-auto text-xs text-slate-500">
          {meta.source} · {meta.bar_count} bars
        </span>
      )}
    </form>
  );
}
