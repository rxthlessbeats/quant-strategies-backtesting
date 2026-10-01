"use client";

import { MoreVertical, X } from "lucide-react";
import {
  colorForSlot,
  paramsDisplay,
  type IndicatorSelection,
} from "@/lib/indicator-utils";

interface ChartLegendProps {
  selections: IndicatorSelection[];
  onOpenSettings: (slotId: string, anchor: HTMLElement) => void;
  onRemove: (slotId: string) => void;
}

export default function ChartLegend({
  selections,
  onOpenSettings,
  onRemove,
}: ChartLegendProps) {
  if (selections.length === 0) return null;

  return (
    <div className="pointer-events-auto absolute left-3 top-3 z-20 max-h-[300px] min-w-[140px] overflow-y-auto rounded-md border border-border bg-card/95 px-2 py-2 backdrop-blur-md">
      <ul className="flex flex-col gap-1.5">
        {selections.map((sel, index) => {
          const color = colorForSlot(index);
          return (
            <li
              key={sel.slotId}
              className="flex items-center gap-2 text-xs text-foreground"
            >
              <span
                className="block h-0.5 w-8 shrink-0 rounded-full"
                style={{ backgroundColor: color }}
                aria-hidden
              />
              <span className="min-w-[2.5rem] lowercase">{sel.id}</span>
              <span className="tabular-nums text-muted-foreground">
                {paramsDisplay(sel.params)}
              </span>
              <button
                type="button"
                onClick={(e) =>
                  onOpenSettings(sel.slotId, e.currentTarget)
                }
                className="rounded p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
                aria-label={`Settings for ${sel.id}`}
              >
                <MoreVertical className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={() => onRemove(sel.slotId)}
                className="rounded p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
                aria-label={`Remove ${sel.id}`}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
