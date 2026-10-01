"use client";

import { Loader2, Plus, Save } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
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
      className="chart-toolbar"
    >
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            ref={addButtonRef}
            type="button"
            variant="ghost"
            size="sm"
            className={cn(
              "h-8 gap-1.5 px-2 text-xs text-muted-foreground",
              "hover:bg-accent hover:text-foreground",
            )}
          >
            <Plus className="h-3.5 w-3.5" />
            Add indicator
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="max-h-[70vh] w-56 overflow-y-auto">
          {catalog.length === 0 ? (
            <DropdownMenuItem disabled>No indicators</DropdownMenuItem>
          ) : (
            Array.from(new Set(catalog.map(item => item.category))).map(category => (
              <DropdownMenuGroup key={category} aria-label={category}>
                <DropdownMenuLabel className="capitalize text-xs text-muted-foreground">{category}</DropdownMenuLabel>
                {catalog.filter(item => item.category === category).map((item) => {
                  const params = Object.fromEntries(
                    Object.entries(item.params)
                      .map(([key, value]) => [key, Number(value)])
                      .filter(([, value]) => Number.isFinite(value)),
                  );
                  return (
                    <DropdownMenuItem
                      key={item.id}
                      title={item.description}
                      onSelect={() => {
                        const anchor = addButtonRef.current;
                        if (!anchor) return;
                        window.setTimeout(() => {
                          onAddIndicatorPick(item.id, params, anchor);
                        }, 0);
                      }}
                    >
                      <span className="uppercase">{item.id}</span>
                    </DropdownMenuItem>
                  );
                })}
                <DropdownMenuSeparator />
              </DropdownMenuGroup>
            ))
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onSaveSettings}
        className={cn(
          "h-8 gap-1.5 px-2 text-xs text-muted-foreground",
          "hover:bg-accent hover:text-foreground",
        )}
      >
        <Save className="h-3.5 w-3.5" />
        Save
      </Button>
      <div className="flex items-center gap-1 rounded-md border border-border bg-muted/50 p-1">
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
              "h-7 px-2 text-[11px] text-muted-foreground",
            selectedView === view && "bg-accent text-foreground",
            )}
            aria-pressed={selectedView === view}
          >
            {view}
          </Button>
        ))}
      </div>
      {loading && (
        <Loader2 className="ml-auto h-4 w-4 animate-spin text-muted-foreground" />
      )}
      {meta && !loading && (
        <span className="ml-auto text-xs text-muted-foreground">
          {meta.source} · {meta.bar_count} bars
        </span>
      )}
    </div>
  );
}
