export const CHART_VIEW_OPTIONS = ["1M", "3M", "6M", "1Y", "5Y", "10Y", "ALL"] as const;

export type ChartView = (typeof CHART_VIEW_OPTIONS)[number];
