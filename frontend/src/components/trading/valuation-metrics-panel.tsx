"use client";

import { formatDailyMarketAsOf } from "@/lib/market-timestamps";
import type { BarPoint, MarketDataAreaResponse } from "@/lib/types";

interface ValuationMetricsPanelProps {
  data: MarketDataAreaResponse | null;
  statements?: MarketDataAreaResponse | null;
  bars: BarPoint[];
  loading: boolean;
  error: string | null;
}

interface YahooValue {
  raw?: unknown;
  fmt?: unknown;
}

function modulePayload(data: MarketDataAreaResponse | null, module: string) {
  return data?.modules.find((item) => item.module === module)?.payload ?? null;
}

function rawNumber(value: unknown): number | null {
  if (value == null) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") {
    const parsed = Number(value.replace(/[$,%]/g, ""));
    return Number.isFinite(parsed) ? parsed : null;
  }
  if (typeof value === "object" && "raw" in value) {
    return rawNumber((value as YahooValue).raw);
  }
  return null;
}

function formattedValue(value: unknown, fallback = "-"): string {
  if (value == null) return fallback;
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
    if (yahooValue.raw != null) return formattedValue(yahooValue.raw, fallback);
  }
  if (typeof value === "number") {
    return new Intl.NumberFormat("en-US", {
      notation: Math.abs(value) >= 1_000_000 ? "compact" : "standard",
      maximumFractionDigits: 2,
    }).format(value);
  }
  return String(value);
}

function hasPercentMetric(value: unknown): boolean {
  const raw = rawNumber(value);
  return raw != null && raw !== 0;
}

function percentOrDash(value: unknown, treatZeroAsMissing = false): string {
  const raw = rawNumber(value);
  if (raw == null || (treatZeroAsMissing && raw === 0)) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 2,
  }).format(raw);
}

function ratioValue(numerator: unknown, denominator: unknown): string {
  const rawNumerator = rawNumber(numerator);
  const rawDenominator = rawNumber(denominator);
  if (rawNumerator == null || rawDenominator == null || rawDenominator === 0) {
    return "-";
  }
  return formattedValue(rawNumerator / rawDenominator);
}

function formatDate(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
    const raw = rawNumber(yahooValue.raw);
    if (raw != null) {
      const ms = raw > 1e12 ? raw : raw * 1000;
      return new Date(ms).toLocaleDateString();
    }
  }
  if (typeof value === "number") {
    const ms = value > 1e12 ? value : value * 1000;
    return new Date(ms).toLocaleDateString();
  }
  const formatted = formattedValue(value);
  return formatted === "-" ? null : formatted;
}

function formatMetricValue(row: MetricRowData): string {
  return typeof row.value === "string" ? row.value : formattedValue(row.value);
}

const MAX_ROWS_PER_COLUMN = 7;

interface MetricRowData {
  label: string;
  value: string | unknown;
  muted?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function fundamentalsSeries(
  data: MarketDataAreaResponse | null,
  type: string,
) {
  const payload = modulePayload(data, "fundamentalsTimeSeriesQuarterly");
  const result = isRecord(payload?.timeseries)
    ? payload.timeseries.result
    : undefined;
  if (!Array.isArray(result)) return [];
  const series = result.find(
    (item) => isRecord(item) && Array.isArray(item[type]),
  );
  if (!isRecord(series) || !Array.isArray(series[type])) return [];
  return series[type].filter(isRecord);
}

function latestSeriesValue(
  data: MarketDataAreaResponse | null,
  types: string[],
) {
  for (const type of types) {
    const values = fundamentalsSeries(data, type);
    const latest = values[values.length - 1];
    const value = rawNumber(latest?.reportedValue);
    if (value != null) return value;
  }
  return null;
}

function trailingSeriesValue(
  data: MarketDataAreaResponse | null,
  type: string,
  count: number,
) {
  const values = fundamentalsSeries(data, type)
    .slice(-count)
    .map((item) => rawNumber(item.reportedValue));
  if (values.length < count || values.some((value) => value == null)) return null;
  return values.reduce<number>((sum, value) => sum + (value ?? 0), 0);
}

function latestQuarterlyCash(data: MarketDataAreaResponse | null) {
  return latestSeriesValue(data, [
    "quarterlyCashAndCashEquivalents",
    "quarterlyCashCashEquivalentsAndShortTermInvestments",
  ]);
}

function trailingTwelveMonthFreeCashFlow(data: MarketDataAreaResponse | null) {
  return trailingSeriesValue(data, "quarterlyFreeCashFlow", 4);
}

function latestBarTimestamp(bars: BarPoint[] | null | undefined): number | null {
  if (!Array.isArray(bars) || bars.length === 0) return null;
  return bars[bars.length - 1]?.timestamp ?? null;
}

function MetricRow({
  label,
  value,
  muted,
}: {
  label: string;
  value: string;
  muted?: string;
}) {
  return (
    <div className="border-b border-dotted border-border/70 py-2.5">
      <div className="flex items-start justify-between gap-4">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-right text-sm font-medium text-foreground">
          {value}
          {muted && (
            <span className="text-xs text-muted-foreground"> ({muted})</span>
          )}
        </span>
      </div>
    </div>
  );
}

function FeaturedMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b border-dotted border-border/70 pb-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold text-foreground">{value}</p>
    </div>
  );
}

export default function ValuationMetricsPanel({
  data,
  statements,
  bars = [],
  loading,
  error,
}: ValuationMetricsPanelProps) {
  const detail = modulePayload(data, "summaryDetail");
  const stats = modulePayload(data, "defaultKeyStatistics");

  const marketCap = detail?.marketCap ?? stats?.marketCap;
  const enterpriseValue = stats?.enterpriseValue;
  const priceToCash = ratioValue(marketCap, latestQuarterlyCash(statements ?? null));
  const priceToFreeCashFlow = ratioValue(
    marketCap,
    trailingTwelveMonthFreeCashFlow(statements ?? null),
  );
  const financial = modulePayload(data, "financialData");
  const dividendRate = detail?.dividendRate;
  const dividendYield =
    detail?.dividendYield ?? stats?.dividendYield ?? financial?.dividendYield;
  const payoutRatio =
    detail?.payoutRatio ?? stats?.payoutRatio ?? financial?.payoutRatio;
  const exDividendDate = formatDate(detail?.exDividendDate);
  const updatedAt = formatDailyMarketAsOf(latestBarTimestamp(bars));
  const hasDividendYield = hasPercentMetric(dividendYield);
  const hasPayoutRatio = hasPercentMetric(payoutRatio);

  const rows: MetricRowData[] = [
    { label: "EPS", value: stats?.trailingEps ?? stats?.forwardEps },
    { label: "Beta", value: detail?.beta ?? stats?.beta },
    {
      label: "Return on Equity",
      value: percentOrDash(financial?.returnOnEquity),
    },
    { label: "Trailing P/E", value: detail?.trailingPE ?? stats?.trailingPE },
    { label: "Forward P/E", value: detail?.forwardPE ?? stats?.forwardPE },
    { label: "PEG ratio", value: stats?.pegRatio },
    { label: "Price / Sales", value: detail?.priceToSalesTrailing12Months },
    { label: "Price / Book", value: stats?.priceToBook },
    { label: "Price / Cash (MRQ)", value: priceToCash },
    { label: "Price / FCF (TTM)", value: priceToFreeCashFlow },
    { label: "EV / EBITDA", value: stats?.enterpriseToEbitda },
    { label: "EV / Sales", value: stats?.enterpriseToRevenue },
    {
      label: "Dividend Yield",
      value: percentOrDash(dividendYield, true),
      muted:
        hasDividendYield && dividendRate != null
          ? `Annual ${formattedValue(dividendRate)}`
          : undefined,
    },
    {
      label: "Payout Ratio",
      value: percentOrDash(payoutRatio, true),
      muted:
        hasPayoutRatio && exDividendDate
          ? `Ex-div ${exDividendDate}`
          : undefined,
    },
  ];
  const columns: MetricRowData[][] = [
    rows.slice(0, MAX_ROWS_PER_COLUMN),
    rows.slice(MAX_ROWS_PER_COLUMN),
  ];

  return (
    <section className="w-full overflow-hidden rounded-lg border border-border bg-black/40 backdrop-blur-md lg:flex-1">
      <div className="flex items-center justify-between border-b border-border/70 px-4 py-3">
        <h2 className="text-sm font-semibold text-foreground">
          Valuation metrics
        </h2>
        {updatedAt && (
          <span className="text-xs text-muted-foreground">
            Updated {updatedAt}
          </span>
        )}
      </div>

      <div className="p-4">
        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        )}

        {!error && loading && (
          <div className="space-y-3">
            {Array.from({ length: 8 }).map((_, index) => (
              <div
                key={index}
                className="h-8 animate-pulse rounded bg-muted/20"
              />
            ))}
          </div>
        )}

        {!error && !loading && (
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <FeaturedMetric
                label="Market Cap"
                value={formattedValue(marketCap)}
              />
              <FeaturedMetric
                label="Enterprise Value"
                value={formattedValue(enterpriseValue)}
              />
            </div>

            <div className="grid gap-x-6 md:grid-cols-2">
              {columns.map((column, columnIndex) => (
                <div key={columnIndex}>
                  {column.map((row: MetricRowData) => (
                    <MetricRow
                      key={row.label}
                      label={row.label}
                      value={formatMetricValue(row)}
                      muted={row.muted}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
