"use client";

import { Loader2, Plus, Save } from "lucide-react";
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
  selectedView: ChartView;
  onViewChange: (view: ChartView) => void;
  loading: boolean;
  meta?: ChartMeta;
  onAddIndicatorPick: (
    id: string,
    defaultParams: Record<string, number>,
    anchor: HTMLElement,
  ) => void;
  onSaveSettings: () => void;
}

export default function ChartToolbar({
  selectedView,
  onViewChange,
  loading,
  meta,
  onAddIndicatorPick,
  onSaveSettings,
}: ChartToolbarProps) {
  const [catalog, setCatalog] = useState<IndicatorCatalogItem[]>([]);
  const addButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    fetchIndicatorCatalog()
      .then(setCatalog)
      .catch(() => setCatalog([]));
  }, []);

  return (
    <div
      className="flex flex-wrap items-center gap-2 px-3 py-2"
    >
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
              const params = Object.fromEntries(
                Object.entries(item.params)
                  .map(([key, value]) => [key, Number(value)])
                  .filter(([, value]) => Number.isFinite(value)),
              );
              return (
                <DropdownMenuItem
                  key={item.id}
                  onSelect={() => {
                    const anchor = addButtonRef.current;
                    if (!anchor) return;
                    window.setTimeout(() => {
                      onAddIndicatorPick(item.id, params, anchor);
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
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onSaveSettings}
        className={cn(
          "h-8 gap-1.5 px-2 text-xs text-slate-300",
          "hover:bg-white/10 hover:text-slate-100",
        )}
      >
        <Save className="h-3.5 w-3.5" />
        Save
      </Button>
      <div className="flex items-center gap-1 rounded-md border border-white/10 bg-black/20 p-1">
        {CHART_VIEW_OPTIONS.map((view) => (
          <Button
            key={view}
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => {
              if (selectedView !== view) {
                onViewChange(view);
              }
            }}
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
    </div>
  );
}
