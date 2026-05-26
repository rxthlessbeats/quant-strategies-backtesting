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
  start: string | null;
  end: string | null;
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

export interface IndexMetricItem {
  id: string;
  label: string;
  symbol: string;
  price: number;
  change: number;
  as_of: string;
}

export interface IndexMetricsResponse {
  metrics: IndexMetricItem[];
}

export interface PerformancePeriodItem {
  id: string;
  label: string;
  symbol_return: number | null;
  benchmark_return: number | null;
}

export interface PerformanceBenchmarkOption {
  symbol: string;
  description: string;
}

export interface PerformanceBenchmarkGroup {
  category: string;
  options: PerformanceBenchmarkOption[];
}

export interface PerformanceBenchmarkOptionsResponse {
  groups: PerformanceBenchmarkGroup[];
}

export interface PerformanceComparisonResponse {
  symbol: string;
  benchmark_label: string;
  benchmark_symbol: string;
  as_of: string;
  periods: PerformancePeriodItem[];
}

export interface TickerSearchItem {
  symbol: string;
  name: string;
  type?: string | null;
  region?: string | null;
  currency?: string | null;
}

export interface TickerSearchResponse {
  results: TickerSearchItem[];
}

export interface CompanyOverview {
  symbol: string;
  asset_type?: string | null;
  name?: string | null;
  description?: string | null;
  cik?: string | null;
  exchange?: string | null;
  currency?: string | null;
  country?: string | null;
  sector?: string | null;
  industry?: string | null;
  address?: string | null;
  fiscal_year_end?: string | null;
  latest_quarter?: string | null;
  ebitda?: string | null;
  book_value?: string | null;
  dividend_per_share?: string | null;
  eps?: string | null;
  revenue_per_share_ttm?: string | null;
  profit_margin?: string | null;
  operating_margin_ttm?: string | null;
  return_on_assets_ttm?: string | null;
  return_on_equity_ttm?: string | null;
  revenue_ttm?: string | null;
  gross_profit_ttm?: string | null;
  diluted_eps_ttm?: string | null;
  quarterly_earnings_growth_yoy?: string | null;
  quarterly_revenue_growth_yoy?: string | null;
  shares_outstanding?: string | null;
  dividend_date?: string | null;
  ex_dividend_date?: string | null;
  analyst_rating_strong_buy?: string | null;
  analyst_rating_buy?: string | null;
  analyst_rating_hold?: string | null;
  analyst_rating_sell?: string | null;
  analyst_rating_strong_sell?: string | null;
  fetched_at?: string | null;
}

export interface HealthResponse {
  status: string;
}
