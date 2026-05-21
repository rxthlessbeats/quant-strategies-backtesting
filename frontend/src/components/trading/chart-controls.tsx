"use client";

import { parseISO } from "date-fns";
import { Loader2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import type { DateRange } from "react-day-picker";
import { Button } from "@/components/ui/button";
import { fetchChart } from "@/lib/api";
import { formatDateParam } from "@/lib/chart-data";
import type { AnalysisChartResponse } from "@/lib/types";
import { DateRangePicker } from "./date-range-picker";
import IndicatorPicker, {
  buildIndicatorsQuery,
  type IndicatorSelection,
} from "./indicator-picker";

interface ChartControlsProps {
  initialSymbol?: string;
  initialStart?: string;
  initialEnd?: string;
  initialIndicators?: string;
  onData: (data: AnalysisChartResponse) => void;
  onLoading?: (loading: boolean) => void;
  onError?: (message: string | null) => void;
}

function parseDateRange(start: string, end: string): DateRange | undefined {
  try {
    return { from: parseISO(start), to: parseISO(end) };
  } catch {
    return undefined;
  }
}

export default function ChartControls({
  initialSymbol = "AAPL",
  initialStart,
  initialEnd,
  initialIndicators = "sma:5,sma:20",
  onData,
  onLoading,
  onError,
}: ChartControlsProps) {
  const [symbol, setSymbol] = useState(initialSymbol);
  const [dateRange, setDateRange] = useState<DateRange | undefined>(() =>
    initialStart && initialEnd
      ? parseDateRange(initialStart, initialEnd)
      : undefined,
  );

  useEffect(() => {
    if (!dateRange?.from && initialStart && initialEnd) {
      setDateRange(parseDateRange(initialStart, initialEnd));
    }
  }, [initialStart, initialEnd, dateRange?.from]);
  const [selections, setSelections] = useState<IndicatorSelection[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!dateRange?.from || !dateRange?.to) {
      onError?.("Select a start and end date.");
      return;
    }
    setLoading(true);
    onLoading?.(true);
    onError?.(null);
    try {
      const indicators =
        buildIndicatorsQuery(selections) || initialIndicators || undefined;
      const data = await fetchChart({
        symbol: symbol.trim().toUpperCase(),
        start: formatDateParam(dateRange.from),
        end: formatDateParam(dateRange.to),
        interval: "1d",
        indicators,
      });
      onData(data);
    } catch (e) {
      onError?.(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
      onLoading?.(false);
    }
  }, [
    dateRange,
    symbol,
    selections,
    initialIndicators,
    onData,
    onLoading,
    onError,
  ]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label className="mb-1 block text-sm text-muted-foreground">
            Symbol
          </label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="w-32 rounded-md border border-input bg-background px-3 py-2 text-sm uppercase"
            placeholder="AAPL"
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-muted-foreground">
            Date range
          </label>
          <DateRangePicker
            dateRange={dateRange}
            onDateRangeChange={setDateRange}
          />
        </div>
        <Button type="button" onClick={load} disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="animate-spin" />
              Loading…
            </>
          ) : (
            "Load chart"
          )}
        </Button>
      </div>
      <IndicatorPicker selections={selections} onChange={setSelections} />
    </div>
  );
}
