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

const PERIOD_ERROR_MESSAGE =
  "Period should be an integer and larger than 1";

function parseIntegerInput(value: string): number | null {
  const trimmed = value.trim();
  if (!/^-?\d+$/.test(trimmed)) return null;
  return parseInt(trimmed, 10);
}

function isValidPeriod(value: string): boolean {
  const parsed = parseIntegerInput(value);
  return parsed !== null && parsed > 1 && parsed <= 500;
}

interface IndicatorSettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: SettingsMode;
  indicatorId: string;
  initialPeriod: number;
  slotId?: string;
  anchorEl: HTMLElement | null;
  selections: IndicatorSelection[];
  onApply: (period: number) => void;
  onBack: () => void;
  onPeriodError?: (message: string) => void;
}

export default function IndicatorSettingsPanel({
  open,
  onOpenChange,
  mode,
  indicatorId,
  initialPeriod,
  slotId,
  anchorEl,
  selections,
  onApply,
  onBack,
  onPeriodError,
}: IndicatorSettingsPanelProps) {
  const [draftPeriodInput, setDraftPeriodInput] = useState(
    String(initialPeriod),
  );
  const [error, setError] = useState<string | null>(null);
  const anchorRef = useRef<HTMLElement | null>(null);
  anchorRef.current = anchorEl;

  const anchorVirtualRef =
    anchorRef as React.RefObject<HTMLElement & Element>;

  useEffect(() => {
    if (open) {
      setDraftPeriodInput(String(initialPeriod));
      setError(null);
    }
  }, [open, initialPeriod, indicatorId, mode, slotId]);

  const handleApply = () => {
    if (!isValidPeriod(draftPeriodInput)) {
      onPeriodError?.(PERIOD_ERROR_MESSAGE);
      return;
    }

    const period = parseIntegerInput(draftPeriodInput)!;
    const isDuplicate = hasDuplicate(selections, indicatorId, period);
    const isSameSlot =
      mode === "edit" &&
      slotId &&
      selections.some(
        (s) => s.slotId === slotId && s.id === indicatorId && s.period === period,
      );

    if (isDuplicate && !isSameSlot) {
      setError("Already on chart");
      return;
    }

    onApply(period);
  };

  const handleBack = () => {
    setDraftPeriodInput(String(initialPeriod));
    setError(null);
    onBack();
  };

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      {anchorEl && <PopoverAnchor virtualRef={anchorVirtualRef as RefObject<Element>} />}
      <PopoverContent
        align="start"
        className="z-[100] w-64 border-white/10 bg-[#0f172a] p-0 text-slate-100"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <div className="border-b border-white/10 px-3 py-2">
          <h3 className="text-sm font-medium">Settings</h3>
        </div>
        <div className="space-y-3 px-3 py-3">
          <div>
            <label className="mb-1 block text-xs text-slate-400">
              Indicator
            </label>
            <span className="text-sm lowercase text-slate-200">
              {indicatorId}
            </span>
          </div>
          <div>
            <label
              htmlFor="indicator-period"
              className="mb-1 block text-xs text-slate-400"
            >
              Period
            </label>
            <input
              id="indicator-period"
              type="text"
              inputMode="numeric"
              value={draftPeriodInput}
              onChange={(e) => {
                setDraftPeriodInput(e.target.value);
                setError(null);
              }}
              className="w-full rounded border border-white/15 bg-black/40 px-2 py-1 text-sm text-slate-100"
            />
          </div>
          {error && (
            <p className="text-xs text-red-400">{error}</p>
          )}
        </div>
        <div className="flex justify-end gap-2 border-t border-white/10 px-3 py-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 text-xs text-slate-300 hover:bg-white/10 hover:text-slate-100"
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
