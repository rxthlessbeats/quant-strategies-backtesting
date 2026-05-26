import type { BarPoint } from "@/lib/types";

export interface LatestQuote {
  price: number;
  changeAmount: number;
  changePercent: number;
  asOf: number;
}

export function latestQuoteFromBars(bars: BarPoint[]): LatestQuote | null {
  const valid = bars.filter(
    (bar) => bar.close !== null && bar.close !== undefined && !Number.isNaN(bar.close),
  );
  if (valid.length === 0) {
    return null;
  }

  const latest = valid[valid.length - 1];
  const previous = valid.length >= 2 ? valid[valid.length - 2] : null;
  const changeAmount = previous ? latest.close - previous.close : 0;
  const changePercent =
    previous && previous.close !== 0 ? changeAmount / previous.close : 0;

  return {
    price: latest.close,
    changeAmount,
    changePercent,
    asOf: latest.timestamp,
  };
}
