export type DataSource = "cache" | "fetch";

export interface BarPoint {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adj_close?: number | null;
}

export interface ChartMeta {
  source: DataSource;
  cached_through?: string | null;
  fetched_at?: string | null;
  bar_count: number;
}

export interface AnalysisChartResponse {
  symbol: string;
  interval: string;
  start: string;
  end: string;
  meta: ChartMeta;
  bars: BarPoint[];
  indicators: Record<string, (number | null)[]>;
}

export interface IndicatorCatalogItem {
  id: string;
  category: string;
  params: Record<string, number | string>;
  description: string;
}

export interface HealthResponse {
  status: string;
}
