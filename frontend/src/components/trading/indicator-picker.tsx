"use client";

import { useEffect, useState } from "react";
import { fetchIndicatorCatalog } from "@/lib/api";
import type { IndicatorCatalogItem } from "@/lib/types";
import { Button } from "@/components/ui/button";

export interface IndicatorSelection {
  key: string;
  id: string;
  period: number;
  enabled: boolean;
}

interface IndicatorPickerProps {
  selections: IndicatorSelection[];
  onChange: (selections: IndicatorSelection[]) => void;
}

export function buildIndicatorsQuery(
  selections: IndicatorSelection[],
): string {
  return selections
    .filter((s) => s.enabled)
    .map((s) => `${s.id}:${s.period}`)
    .join(",");
}

export default function IndicatorPicker({
  selections,
  onChange,
}: IndicatorPickerProps) {
  const [catalog, setCatalog] = useState<IndicatorCatalogItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchIndicatorCatalog()
      .then((items) => {
        setCatalog(items);
        if (selections.length === 0) {
          const initial: IndicatorSelection[] = [
            { key: "sma:5", id: "sma", period: 5, enabled: true },
            { key: "sma:20", id: "sma", period: 20, enabled: true },
            {
              key: "ema:50",
              id: "ema",
              period:
                typeof items.find((i) => i.id === "ema")?.params.period ===
                "number"
                  ? (items.find((i) => i.id === "ema")!.params.period as number)
                  : 50,
              enabled: false,
            },
          ];
          onChange(initial);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- init once
  }, []);

  const toggle = (key: string) => {
    onChange(
      selections.map((s) =>
        s.key === key ? { ...s, enabled: !s.enabled } : s,
      ),
    );
  };

  const setPeriod = (key: string, period: number) => {
    onChange(
      selections.map((s) =>
        s.key === key ? { ...s, period, key: `${s.id}:${period}` } : s,
      ),
    );
  };

  if (error) {
    return <p className="text-sm text-destructive">{error}</p>;
  }

  if (selections.length === 0) {
    return <p className="text-sm text-muted-foreground">Loading indicators…</p>;
  }

  return (
    <div className="flex flex-wrap gap-3">
      {selections.map((sel) => {
        const meta = catalog.find((c) => c.id === sel.id);
        return (
          <div
            key={sel.key}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-2"
          >
            <Button
              type="button"
              size="sm"
              variant={sel.enabled ? "default" : "outline"}
              onClick={() => toggle(sel.key)}
            >
              {sel.id.toUpperCase()} {sel.period}
            </Button>
            <label className="flex items-center gap-1 text-sm text-muted-foreground">
              period
              <input
                type="number"
                min={1}
                max={500}
                value={sel.period}
                disabled={!sel.enabled}
                onChange={(e) =>
                  setPeriod(sel.key, parseInt(e.target.value, 10) || 1)
                }
                className="w-16 rounded border border-input bg-background px-2 py-1 text-foreground"
              />
            </label>
            {meta && (
              <span className="hidden text-xs text-muted-foreground lg:inline">
                {meta.description}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
