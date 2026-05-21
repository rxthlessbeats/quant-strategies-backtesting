import type { BarPoint } from "@/lib/types";
import type {
  CandlestickData,
  HistogramData,
  LineData,
  UTCTimestamp,
} from "lightweight-charts";

export function toCandlestickData(bars: BarPoint[]): CandlestickData<UTCTimestamp>[] {
  return bars.map((bar) => ({
    time: bar.timestamp as UTCTimestamp,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
  }));
}

export function toVolumeData(bars: BarPoint[]): HistogramData<UTCTimestamp>[] {
  return bars.map((bar) => ({
    time: bar.timestamp as UTCTimestamp,
    value: bar.volume,
    color:
      bar.close >= bar.open
        ? "rgba(38, 166, 154, 0.5)"
        : "rgba(239, 83, 80, 0.5)",
  }));
}

export function toIndicatorLineData(
  bars: BarPoint[],
  values: (number | null)[],
): LineData<UTCTimestamp>[] {
  const points: LineData<UTCTimestamp>[] = [];
  const len = Math.min(bars.length, values.length);
  for (let i = 0; i < len; i++) {
    const v = values[i];
    if (v != null && Number.isFinite(v)) {
      points.push({
        time: bars[i].timestamp as UTCTimestamp,
        value: v,
      });
    }
  }
  return points;
}

export const INDICATOR_COLORS = [
  "#2962FF",
  "#E91E63",
  "#FF9800",
  "#9C27B0",
  "#00BCD4",
  "#4CAF50",
];

export function formatDateParam(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export function defaultDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 1);
  return { start: formatDateParam(start), end: formatDateParam(end) };
}
