export const CHART_VIEW_OPTIONS = ["1M", "3M", "6M", "1Y", "5Y", "10Y", "ALL"] as const;

export type ChartView = (typeof CHART_VIEW_OPTIONS)[number];

export function chartViewStart(timestamp: number, view: ChartView): number {
  if (view === "ALL") return 0;
  const months = { "1M": 1, "3M": 3, "6M": 6, "1Y": 12, "5Y": 60, "10Y": 120 }[view];
  const date = new Date(timestamp * 1000);
  const day = date.getUTCDate();
  date.setUTCDate(1);
  date.setUTCMonth(date.getUTCMonth() - months);
  const lastDay = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + 1, 0)).getUTCDate();
  date.setUTCDate(Math.min(day, lastDay));
  return Math.floor(date.getTime() / 1000);
}
