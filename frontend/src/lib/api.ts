import type {
  AnalysisChartResponse,
  CompanyOverview,
  HealthResponse,
  IndexMetricsResponse,
  IndicatorCatalogItem,
  PerformanceBenchmarkOptionsResponse,
  PerformanceComparisonResponse,
  TickerSearchResponse,
} from "@/lib/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
  return API_BASE;
}

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export interface ChartQueryParams {
  symbol: string;
  start?: string;
  end?: string;
  interval?: string;
  indicators?: string;
}

export function fetchChart(
  params: ChartQueryParams,
): Promise<AnalysisChartResponse> {
  const search = new URLSearchParams({
    symbol: params.symbol,
    interval: params.interval ?? "1d",
  });
  if (params.start) {
    search.set("start", params.start);
  }
  if (params.end) {
    search.set("end", params.end);
  }
  if (params.indicators?.trim()) {
    search.set("indicators", params.indicators.trim());
  }
  return apiFetch(`/api/v1/analysis/chart?${search.toString()}`);
}

export function fetchIndicatorCatalog(): Promise<IndicatorCatalogItem[]> {
  return apiFetch("/api/v1/analysis/indicators");
}

export function fetchIndexMetrics(): Promise<IndexMetricsResponse> {
  return apiFetch("/api/v1/market/index-metrics");
}

export function fetchPerformanceBenchmarkOptions(): Promise<PerformanceBenchmarkOptionsResponse> {
  return apiFetch("/api/v1/market/performance-benchmarks");
}

export function fetchPerformanceComparison(
  symbol: string,
  benchmark = "SPY",
): Promise<PerformanceComparisonResponse> {
  const search = new URLSearchParams({ benchmark });
  return apiFetch(
    `/api/v1/market/performance-comparison/${encodeURIComponent(symbol)}?${search.toString()}`,
  );
}

export function searchTickers(keywords: string): Promise<TickerSearchResponse> {
  const search = new URLSearchParams({ keywords });
  return apiFetch(`/api/v1/market/search?${search.toString()}`);
}

export function fetchCompanyOverview(symbol: string): Promise<CompanyOverview> {
  return apiFetch(`/api/v1/market/overview/${encodeURIComponent(symbol)}`);
}

export function fetchHealth(): Promise<HealthResponse> {
  return apiFetch("/health");
}
