import type {
  AnalysisChartResponse,
  CompanyOverview,
  HealthResponse,
  IndexMetricsResponse,
  IndicatorCatalogItem,
  MarketDataAreaResponse,
  PerformanceBenchmarkOptionsResponse,
  PerformanceComparisonResponse,
  TickerSearchResponse,
} from "@/lib/types";

const BACKEND_URL = (
  process.env.API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

export function getApiBaseUrl(): string {
  return "/api/backend";
}

function fetchTarget(path: string): { url: string; headers: HeadersInit } {
  if (typeof window === "undefined") {
    const headers: HeadersInit = {};
    if (process.env.API_SECRET) {
      headers["X-API-Key"] = process.env.API_SECRET;
    }
    return { url: `${BACKEND_URL}${path}`, headers };
  }
  return { url: `/api/backend${path}`, headers: {} };
}

async function apiFetch<T>(path: string, attempt = 0): Promise<T> {
  const { url, headers } = fetchTarget(path);
  const res = await fetch(url, { cache: "no-store", headers });
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
    const retryable = res.status === 502 || res.status === 503;
    if (attempt < 1 && retryable) {
      await new Promise((resolve) => setTimeout(resolve, 800));
      return apiFetch<T>(path, attempt + 1);
    }
    throw new Error(publicApiError(detail));
  }
  return res.json() as Promise<T>;
}

function publicApiError(detail: string): string {
  const lowered = detail.toLowerCase();
  if (
    lowered.includes("[sql:") ||
    lowered.includes("sqlite") ||
    lowered.includes("sqlalchemy") ||
    lowered.includes("operationalerror") ||
    lowered.includes("pendingrollback")
  ) {
    return "Fetching this ticker for the first time. Please wait a moment.";
  }
  return detail;
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

export function fetchMarketDataArea(
  symbol: string,
  area: string,
  options?: { force?: boolean },
): Promise<MarketDataAreaResponse> {
  const query = options?.force ? "?force=true" : "";
  return apiFetch(
    `/api/v1/market/data/${encodeURIComponent(symbol)}/areas/${encodeURIComponent(area)}${query}`,
  );
}

export function fetchHealth(): Promise<HealthResponse> {
  return apiFetch("/health");
}
