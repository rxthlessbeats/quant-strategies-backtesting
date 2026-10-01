"use client";

import { VChart } from "@visactor/react-vchart";
import { ChartThemeProvider } from "@/components/providers/chart-theme-provider";
import type { IBarChartSpec } from "@visactor/vchart";
import { useMemo, useState } from "react";
import type { BarPoint, MarketDataAreaResponse } from "@/lib/types";

interface AnalystRecommendationsPanelProps {
  data: MarketDataAreaResponse | null;
  marketStats: MarketDataAreaResponse | null;
  bars: BarPoint[];
  loading: boolean;
  error: string | null;
}

interface TrendItem {
  period?: unknown;
  strongBuy?: unknown;
  buy?: unknown;
  hold?: unknown;
  sell?: unknown;
  strongSell?: unknown;
}

interface HistoryItem {
  epochGradeDate?: unknown;
  firm?: unknown;
  toGrade?: unknown;
  fromGrade?: unknown;
  action?: unknown;
  priceTargetAction?: unknown;
  currentPriceTarget?: unknown;
  priorPriceTarget?: unknown;
}

interface YahooValue {
  raw?: unknown;
  fmt?: unknown;
}

interface TargetMetrics {
  price: number | null;
  low: number | null;
  mean: number | null;
  high: number | null;
}

const RECOMMENDATION_BUCKETS = [
  { key: "strongBuy", label: "Strong Buy", color: "#14775d" },
  { key: "buy", label: "Buy", color: "#39966c" },
  { key: "hold", label: "Hold", color: "#b9872b" },
  { key: "sell", label: "Sell", color: "#cb743e" },
  { key: "strongSell", label: "Strong Sell", color: "#be3557" },
] as const;

const STACKED_RECOMMENDATION_BUCKETS = [...RECOMMENDATION_BUCKETS].reverse();
const LEGEND_RECOMMENDATION_BUCKETS = RECOMMENDATION_BUCKETS;

function modulePayload(data: MarketDataAreaResponse | null, module: string) {
  return data?.modules.find((item) => item.module === module)?.payload ?? null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
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

function textValue(value: unknown): string {
  if (value == null || value === "") return "-";
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
    if (yahooValue.raw != null) return textValue(yahooValue.raw);
  }
  return String(value);
}

function formatPriceTarget(value: unknown): string {
  const raw = rawNumber(value);
  if (raw == null) return "-";
  return raw.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatDate(value: unknown): string {
  const raw = rawNumber(value);
  if (raw == null) return "-";
  const ms = raw > 1e12 ? raw : raw * 1000;
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function gradeScore(value: string): number | null {
  const normalized = value.toLowerCase().replace(/[_-]/g, " ");
  if (
    normalized.includes("strong buy") ||
    normalized.includes("top pick")
  ) {
    return 5;
  }
  if (
    normalized.includes("buy") ||
    normalized.includes("outperform") ||
    normalized.includes("overweight") ||
    normalized.includes("positive")
  ) {
    return 4;
  }
  if (
    normalized.includes("hold") ||
    normalized.includes("neutral") ||
    normalized.includes("market perform") ||
    normalized.includes("sector perform") ||
    normalized.includes("equal weight")
  ) {
    return 3;
  }
  if (
    normalized.includes("underperform") ||
    normalized.includes("underweight") ||
    normalized.includes("reduce")
  ) {
    return 2;
  }
  if (normalized.includes("sell") || normalized.includes("negative")) {
    return 1;
  }
  return null;
}

function ratingAction(fromGrade: string, toGrade: string): string {
  if (toGrade === "-") return "-";
  if (fromGrade === "-" || fromGrade === toGrade) return "Maintain";
  const fromScore = gradeScore(fromGrade);
  const toScore = gradeScore(toGrade);
  if (fromScore == null || toScore == null) return "Change";
  if (toScore > fromScore) return "Upgrade";
  if (toScore < fromScore) return "Downgrade";
  return "Maintain";
}

function ratingLabel(fromGrade: string, toGrade: string): string {
  if (toGrade === "-") return "-";
  if (fromGrade === "-" || fromGrade === toGrade) return toGrade;
  return `${fromGrade} -> ${toGrade}`;
}

function periodLabel(value: unknown): string {
  const text = textValue(value);
  if (text === "0m") return "Current";
  if (text.startsWith("-") && text.endsWith("m")) {
    return `${text.slice(1, -1)}M ago`;
  }
  return text;
}

function trendItems(data: MarketDataAreaResponse | null): TrendItem[] {
  const payload = modulePayload(data, "recommendationTrend");
  const trend = isRecord(payload) ? payload.trend : undefined;
  return Array.isArray(trend) ? trend.filter(isRecord) : [];
}

function historyItems(data: MarketDataAreaResponse | null): HistoryItem[] {
  const payload = modulePayload(data, "upgradeDowngradeHistory");
  const history = isRecord(payload) ? payload.history : undefined;
  if (!Array.isArray(history)) return [];
  return history
    .filter(isRecord)
    .sort(
      (a, b) =>
        (rawNumber(b.epochGradeDate) ?? 0) - (rawNumber(a.epochGradeDate) ?? 0),
    );
}

function latestPrice(bars: BarPoint[] | null | undefined): number | null {
  if (!Array.isArray(bars) || bars.length === 0) return null;
  const latest = bars[bars.length - 1];
  return latest?.close ?? null;
}

function targetMetrics(
  data: MarketDataAreaResponse | null,
  bars: BarPoint[] | null | undefined,
) {
  const financial = modulePayload(data, "financialData");
  const detail = modulePayload(data, "summaryDetail");
  const price =
    latestPrice(bars) ??
    rawNumber(financial?.currentPrice) ??
    rawNumber(detail?.regularMarketPreviousClose) ??
    rawNumber(detail?.previousClose);

  return {
    price,
    low: rawNumber(financial?.targetLowPrice),
    mean: rawNumber(financial?.targetMeanPrice),
    high: rawNumber(financial?.targetHighPrice),
  };
}

function bucketValue(item: TrendItem | null, key: keyof TrendItem): number {
  return rawNumber(item?.[key]) ?? 0;
}

function rangePosition(
  low: number | null,
  high: number | null,
  value: number | null,
): number {
  if (low == null || high == null || value == null || high <= low) return 0;
  return Math.min(1, Math.max(0, (value - low) / (high - low)));
}

export default function AnalystRecommendationsPanel({
  data,
  marketStats,
  bars = [],
  loading,
  error,
}: AnalystRecommendationsPanelProps) {
  const trends = useMemo(() => trendItems(data), [data]);
  const history = useMemo(() => historyItems(data).slice(0, 8), [data]);
  const targets = useMemo(
    () => targetMetrics(marketStats, bars),
    [marketStats, bars],
  );
  const hasTargetChart = Object.values(targets).some((value) => value != null);

  return (
    <ChartThemeProvider>
      <section className="research-section analyst-section" aria-busy={loading}>
        <div className="research-heading"><div><h2>Analyst Recommendations</h2><p>Price targets, conviction, and the latest rating changes.</p></div></div>
        {error ? <p className="research-error">{error}</p> : loading ? <LoadingState /> : trends.length || history.length || hasTargetChart ? (
          <>
            <div className="analyst-outlook">
              <TargetPriceRange targets={targets} />
              {trends.length > 1 ? <RecommendationStackedBar trends={trends.slice(0, 4).reverse()} /> : (
                <div className="recommendation-history"><h3>Recommendation history</h3><p className="research-empty">No recommendation history available.</p></div>
              )}
            </div>
            <div className="analyst-updates"><RecentRatingActions history={history} /><CurrentMonthPriceChanges history={history} /></div>
          </>
        ) : <p className="research-empty">No analyst data available.</p>}
      </section>
    </ChartThemeProvider>
  );
}

function TargetPriceRange({ targets }: { targets: TargetMetrics }) {
  const [focus, setFocus] = useState<"mean" | "price">("mean");
  const hasRange = targets.low != null && targets.high != null && targets.high > targets.low;
  const pricePosition = rangePosition(targets.low, targets.high, targets.price);
  const meanPosition = rangePosition(targets.low, targets.high, targets.mean);

  return (
    <div className="analyst-target">
      <div className="target-heading"><h3>Price target range</h3><span>USD</span></div>
      <div className="target-selector" role="group" aria-label="Price target focus">
        <button type="button" aria-pressed={focus === "mean"} onClick={() => setFocus("mean")}>Mean target</button>
        <button type="button" aria-pressed={focus === "price"} onClick={() => setFocus("price")}>Current price</button>
      </div>
      <p key={focus} className="target-focus-value" aria-live="polite" aria-atomic="true"><span className="sr-only">{focus === "mean" ? "Mean target" : "Current price"}: </span>{formatPriceTarget(targets[focus])}</p>
      {hasRange ? (
        <div className="target-range">
          <div className="target-track" aria-hidden="true">
            {targets.price != null && <span className="target-marker current-marker" data-active={focus === "price"} style={{ left: `${pricePosition * 100}%` }} />}
            {targets.mean != null && <span className="target-marker mean-marker" data-active={focus === "mean"} style={{ left: `${meanPosition * 100}%` }} />}
          </div>
          <dl className="target-extents"><div><dt>Low target</dt><dd>{formatPriceTarget(targets.low)}</dd></div><div><dt>High target</dt><dd>{formatPriceTarget(targets.high)}</dd></div></dl>
        </div>
      ) : <p className="research-empty">No price target range available.</p>}
      <dl className="target-key-values"><div><dt><span className="series-key current-key" />Current price</dt><dd>{formatPriceTarget(targets.price)}</dd></div><div><dt><span className="series-key company-key" />Mean target</dt><dd>{formatPriceTarget(targets.mean)}</dd></div></dl>
    </div>
  );
}

function RecommendationStackedBar({ trends }: { trends: TrendItem[] }) {
  const values = useMemo(
    () =>
      trends.flatMap((item) =>
        STACKED_RECOMMENDATION_BUCKETS.map((bucket) => ({
          period: periodLabel(item.period),
          type: bucket.label,
          count: bucketValue(item, bucket.key),
        })),
      ),
    [trends],
  );

  const spec = useMemo<IBarChartSpec>(() => ({
    type: "bar",
    animation: typeof window !== "undefined" && !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    data: [
      {
        id: "recommendationTrendData",
        values,
      },
    ],
    xField: "period",
    yField: "count",
    seriesField: "type",
    stack: true,
    height: 230,
    padding: [12, 8, 8, 0],
    color: STACKED_RECOMMENDATION_BUCKETS.map((bucket) => bucket.color),
    legends: {
      visible: false,
    },
    tooltip: {
      trigger: ["click", "hover"],
    },
    axes: [
      {
        orient: "left",
        label: {
        },
        grid: {
          visible: true,
        },
      },
      {
        orient: "bottom",
        label: {
        },
      },
    ],
    bar: {
      state: {
        hover: {
          outerBorder: {
            distance: 2,
            lineWidth: 2,
          },
        },
      },
      style: {
        cornerRadius: [4, 4, 0, 0],
      },
    },
  }), [values]);

  return (
    <div className="recommendation-history">
      <h3>Recommendation history</h3>
      <div className="recommendation-plot">
        <div className="min-w-0 flex-1">
          <VChart spec={spec} />
        </div>
        <div className="recommendation-legend">
          {LEGEND_RECOMMENDATION_BUCKETS.map((bucket) => (
            <div key={bucket.key} className="flex items-center gap-2 text-xs text-muted-foreground">
              <span
                className="h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: bucket.color }}
              />
              <span>{bucket.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RecentRatingActions({ history }: { history: HistoryItem[] }) {
  const latest = history[0] ?? null;

  return (
    <div className="rating-update">
      <h3>Latest rating action</h3>
      {latest ? (
        <RatingAction item={latest} />
      ) : (
        <p className="px-3 py-4 text-sm text-muted-foreground">No recent actions.</p>
      )}
    </div>
  );
}

function CurrentMonthPriceChanges({ history }: { history: HistoryItem[] }) {
  const latestTimestamp = rawNumber(history[0]?.epochGradeDate);
  const latestDate =
    latestTimestamp == null
      ? null
      : new Date((latestTimestamp > 1e12 ? latestTimestamp : latestTimestamp * 1000));
  const currentMonthItems =
    latestDate == null
      ? []
      : history.filter((item) => {
          const timestamp = rawNumber(item.epochGradeDate);
          if (timestamp == null) return false;
          const date = new Date((timestamp > 1e12 ? timestamp : timestamp * 1000));
          return (
            date.getUTCFullYear() === latestDate.getUTCFullYear() &&
            date.getUTCMonth() === latestDate.getUTCMonth()
          );
        });
  const counts = currentMonthItems.reduce(
    (acc, item) => {
      const action = textValue(item.priceTargetAction).toLowerCase();
      if (action.includes("raise")) acc.raises += 1;
      else if (action.includes("lower")) acc.lowers += 1;
      else if (action.includes("maintain")) acc.maintains += 1;
      else acc.other += 1;
      return acc;
    },
    { raises: 0, maintains: 0, lowers: 0, other: 0 },
  );
  const total =
    counts.raises + counts.maintains + counts.lowers + counts.other;
  const monthLabel =
    latestDate == null
      ? "-"
      : latestDate.toLocaleDateString("en-US", {
          month: "short",
          year: "numeric",
          timeZone: "UTC",
        });

  return (
    <div className="target-changes">
      <h3>Price target changes</h3>
      {total > 0 ? (
        <div className="target-change-content">
          <div className="mb-3 flex items-center justify-between text-xs">
            <span className="text-muted-foreground">{monthLabel}</span>
            <span className="font-medium text-foreground">{total} updates</span>
          </div>
          <div className="flex h-3 overflow-hidden rounded-full bg-muted">
            <DistributionSegment
              count={counts.raises}
              total={total}
              color="#14775d"
              label="Raises"
            />
            <DistributionSegment
              count={counts.maintains}
              total={total}
              color="#b9872b"
              label="Maintains"
            />
            <DistributionSegment
              count={counts.lowers}
              total={total}
              color="#be3557"
              label="Lowers"
            />
            <DistributionSegment
              count={counts.other}
              total={total}
              color="#64748b"
              label="Other"
            />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <DistributionRow label="Raises" count={counts.raises} color="#14775d" />
            <DistributionRow
              label="Maintains"
              count={counts.maintains}
              color="#b9872b"
            />
            <DistributionRow label="Lowers" count={counts.lowers} color="#be3557" />
            <DistributionRow label="Other" count={counts.other} color="#64748b" />
          </div>
        </div>
      ) : (
        <p className="px-3 py-4 text-sm text-muted-foreground">
          No current month price target changes.
        </p>
      )}
    </div>
  );
}

function DistributionSegment({
  count,
  total,
  color,
  label,
}: {
  count: number;
  total: number;
  color: string;
  label: string;
}) {
  if (!count) return null;
  return (
    <div
      style={{ width: `${(count / total) * 100}%`, backgroundColor: color }}
      title={`${label}: ${count}`}
    />
  );
}

function DistributionRow({
  label,
  count,
  color,
}: {
  label: string;
  count: number;
  color: string;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="flex items-center gap-1.5 text-muted-foreground">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        {label}
      </span>
      <span className="font-medium text-foreground">{count}</span>
    </div>
  );
}

function RatingAction({ item }: { item: HistoryItem }) {
  const priorTarget = rawNumber(item.priorPriceTarget);
  const currentTarget = rawNumber(item.currentPriceTarget);
  const targetChange =
    priorTarget != null && currentTarget != null ? currentTarget - priorTarget : null;
  const fromGrade = textValue(item.fromGrade);
  const toGrade = textValue(item.toGrade);
  const action = ratingAction(fromGrade, toGrade);
  const rating = ratingLabel(fromGrade, toGrade);

  return (
    <div className="rating-action">
      <div className="flex justify-between gap-3">
        <span className="text-muted-foreground">Date</span>
        <span className="text-right text-foreground">
          {formatDate(item.epochGradeDate)}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-muted-foreground">Analyst</span>
        <span className="text-right font-medium text-foreground">
          {textValue(item.firm)}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-muted-foreground">Rating action</span>
        <span className="text-right text-foreground">
          {action}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-muted-foreground">Rating</span>
        <span className="text-right text-foreground">
          {rating}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-muted-foreground">Price Target</span>
        <span
          className={`text-right font-medium ${
            targetChange == null
              ? "text-foreground"
              : targetChange >= 0
                ? "text-[var(--positive)]"
                : "text-[var(--negative)]"
          }`}
        >
          {priorTarget == null || currentTarget == null
            ? formatPriceTarget(item.currentPriceTarget)
            : `${formatPriceTarget(priorTarget)} -> ${formatPriceTarget(currentTarget)}`}
        </span>
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="analyst-loading">
      <div className="h-40 animate-pulse rounded-md border border-border bg-muted/20" />
      <div className="h-48 animate-pulse rounded-md border border-border bg-muted/20" />
      <div className="h-64 animate-pulse rounded-md border border-border bg-muted/20" />
      <div className="h-64 animate-pulse rounded-md border border-border bg-muted/20" />
    </div>
  );
}
