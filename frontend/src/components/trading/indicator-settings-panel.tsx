"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/ui/popover";
import { hasDuplicate, type IndicatorSelection } from "@/lib/indicator-utils";

export type SettingsMode = "add" | "edit";

const PARAM_ERROR_MESSAGE =
  "Indicator settings should be positive numbers no larger than 500. Bar periods must be whole numbers.";

function parseNumberInput(value: string): number | null {
  const trimmed = value.trim();
  if (!/^-?\d+(\.\d+)?$/.test(trimmed)) return null;
  return Number(trimmed);
}

function isValidParam(key: string, value: string): boolean {
  const parsed = parseNumberInput(value);
  return parsed !== null && parsed > 0 && parsed <= 500 && (key === "std" || Number.isInteger(parsed));
}

function paramsEqual(
  left: Record<string, number>,
  right: Record<string, number>,
): boolean {
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key, index) => key === rightKeys[index] && left[key] === right[key],
    )
  );
}

interface IndicatorSettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: SettingsMode;
  indicatorId: string;
  initialParams: Record<string, number>;
  slotId?: string;
  anchorEl: HTMLElement | null;
  selections: IndicatorSelection[];
  onApply: (params: Record<string, number>) => void;
  onBack: () => void;
  onPeriodError?: (message: string) => void;
}

export default function IndicatorSettingsPanel({
  open,
  onOpenChange,
  mode,
  indicatorId,
  initialParams,
  slotId,
  anchorEl,
  selections,
  onApply,
  onBack,
  onPeriodError,
}: IndicatorSettingsPanelProps) {
  const [draftInputs, setDraftInputs] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const anchorRef = useRef<HTMLElement | null>(null);
  anchorRef.current = anchorEl;

  const anchorVirtualRef =
    anchorRef as React.RefObject<HTMLElement & Element>;

  useEffect(() => {
    if (open) {
      setDraftInputs(
        Object.fromEntries(
          Object.entries(initialParams).map(([key, value]) => [
            key,
            String(value),
          ]),
        ),
      );
      setError(null);
    }
  }, [open, initialParams, indicatorId, mode, slotId]);

  const handleApply = () => {
    const entries = Object.entries(draftInputs);
    if (entries.some(([key, value]) => !isValidParam(key, value))) {
      onPeriodError?.(PARAM_ERROR_MESSAGE);
      return;
    }

    const params = Object.fromEntries(
      entries.map(([key, value]) => [key, parseNumberInput(value)!]),
    );
    if (params.fast !== undefined && params.fast >= params.slow) {
      setError("Fast must be smaller than slow");
      return;
    }
    const isDuplicate = hasDuplicate(selections, indicatorId, params);
    const isSameSlot =
      mode === "edit" &&
      slotId &&
      selections.some(
        (s) =>
          s.slotId === slotId &&
          s.id === indicatorId &&
          paramsEqual(s.params, params),
      );

    if (isDuplicate && !isSameSlot) {
      setError("Already on chart");
      return;
    }

    onApply(params);
  };

  const handleBack = () => {
    setDraftInputs(
      Object.fromEntries(
        Object.entries(initialParams).map(([key, value]) => [key, String(value)]),
      ),
    );
    setError(null);
    onBack();
  };

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      {anchorEl && <PopoverAnchor virtualRef={anchorVirtualRef as RefObject<Element>} />}
      <PopoverContent
        align="start"
        className="z-[100] w-64 border-border bg-popover p-0 text-foreground"
        onCloseAutoFocus={(e) => { e.preventDefault(); anchorEl?.focus(); }}
      >
        <div className="border-b border-border px-3 py-2">
          <h3 className="text-sm font-medium">Settings</h3>
        </div>
        <div className="space-y-3 px-3 py-3">
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">
              Indicator
            </label>
            <span className="text-sm lowercase text-foreground">
              {indicatorId}
            </span>
          </div>
          {Object.entries(draftInputs).map(([key, value]) => (
            <div key={key}>
              <label
                htmlFor={`indicator-${key}`}
                className="mb-1 block text-xs capitalize text-muted-foreground"
              >
                {key === "std" ? "Standard deviations" : (key.charAt(0).toUpperCase() + key.slice(1)).replaceAll("_", " ")}
              </label>
              <input
                id={`indicator-${key}`}
                type="text"
                inputMode="decimal"
                value={value}
                onChange={(e) => {
                  setDraftInputs((prev) => ({
                    ...prev,
                    [key]: e.target.value,
                  }));
                  setError(null);
                }}
                className="w-full rounded border border-border bg-card px-2 py-1 text-sm text-foreground"
              />
            </div>
          ))}
          {Object.keys(draftInputs).length === 0 && <p className="text-xs text-muted-foreground">No parameters needed. Apply to add this indicator.</p>}
          {error && (
            <p className="text-xs text-red-400">{error}</p>
          )}
        </div>
        <div className="flex justify-end gap-2 border-t border-border px-3 py-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={handleBack}
          >
            Back
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-8 text-xs"
            onClick={handleApply}
          >
            Apply
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
