"use client";

import { VChart } from "@visactor/react-vchart";
import type { IBarChartSpec } from "@visactor/vchart";
import { useMemo } from "react";
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
  { key: "strongBuy", label: "Strong Buy", color: "#10b981" },
  { key: "buy", label: "Buy", color: "#22c55e" },
  { key: "hold", label: "Hold", color: "#f59e0b" },
  { key: "sell", label: "Sell", color: "#f97316" },
  { key: "strongSell", label: "Strong Sell", color: "#ef4444" },
] as const;

const TARGET_SERIES = [
  { key: "price", label: "Now Price", color: "#38bdf8" },
  { key: "low", label: "Low Target", color: "#f97316" },
  { key: "mean", label: "Mean Target", color: "#a78bfa" },
  { key: "high", label: "High Target", color: "#10b981" },
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
    <section className="rounded-lg border border-border bg-black/40 px-3 py-3 backdrop-blur-md">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-medium text-slate-200">
          Analyst Recommendations
        </h2>
      </div>

      {error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : loading ? (
        <LoadingState />
      ) : trends.length || history.length || hasTargetChart ? (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
          <TargetPriceRange targets={targets} />
          {trends.length > 1 ? (
            <RecommendationStackedBar trends={trends.slice(0, 4).reverse()} />
          ) : (
            <div className="rounded-md border border-white/10 p-3">
              <p className="text-sm font-medium text-slate-100">Recommendation history</p>
              <p className="py-8 text-sm text-slate-500">
                No recommendation history available.
              </p>
            </div>
          )}
          <RecentRatingActions history={history} />
          <CurrentMonthPriceChanges history={history} />
        </div>
      ) : (
        <p className="text-sm text-slate-500">No analyst data available.</p>
      )}
    </section>
  );
}

function TargetPriceRange({ targets }: { targets: TargetMetrics }) {
  const hasRange = targets.low != null && targets.high != null && targets.high > targets.low;
  const pricePosition = rangePosition(targets.low, targets.high, targets.price);
  const meanPosition = rangePosition(targets.low, targets.high, targets.mean);
  const priceColor = TARGET_SERIES.find((series) => series.key === "price")?.color;
  const meanColor = TARGET_SERIES.find((series) => series.key === "mean")?.color;

  return (
    <div className="rounded-md border border-white/10 p-3">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-slate-100">Price target range</p>
        </div>
      </div>
      {hasRange ? (
        <>
          <div className="relative mt-12 h-10">
            <div className="absolute inset-x-0 top-7 h-3 rounded-full bg-slate-700/80" />
            <div
              className="absolute top-[-5px] z-10 flex -translate-x-1/2 flex-col items-center"
              style={{ left: `${pricePosition * 100}%` }}
              title={`Now price ${formatPriceTarget(targets.price)}`}
            >
              <span className="whitespace-nowrap rounded border border-sky-400/70 px-1.5 py-0.5 text-xs font-medium text-sky-300">
                Now {formatPriceTarget(targets.price)}
              </span>
              <span
                className="h-6 w-0.5"
                style={{ backgroundColor: priceColor }}
              />
            </div>
            <div
              className="absolute top-7 z-10 flex -translate-x-1/2 flex-col items-center"
              style={{ left: `${meanPosition * 100}%` }}
              title={`Mean target ${formatPriceTarget(targets.mean)}`}
            >
              <span
                className="h-6 w-0.5"
                style={{ backgroundColor: meanColor }}
              />
              <span className="mt-1 whitespace-nowrap rounded border border-violet-400/70 px-1.5 py-0.5 text-xs font-medium text-violet-300">
                Mean {formatPriceTarget(targets.mean)}
              </span>
            </div>
          </div>
          <div className="flex justify-between gap-2 text-xs text-slate-500">
            <span>Low {formatPriceTarget(targets.low)}</span>
            <span>High {formatPriceTarget(targets.high)}</span>
          </div>
        </>
      ) : (
        <p className="py-6 text-sm text-slate-500">
          No price target data available.
        </p>
      )}
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
    height: 180,
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
          style: {
            fill: "#94a3b8",
          },
        },
        grid: {
          visible: true,
          style: {
            stroke: "#1f2937",
          },
        },
      },
      {
        orient: "bottom",
        label: {
          style: {
            fill: "#94a3b8",
          },
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
    <div className="rounded-md border border-white/10 p-3">
      <div className="mb-3">
        <p className="text-sm font-medium text-slate-100">Recommendation history</p>
      </div>
      <div className="flex h-44 gap-3">
        <div className="min-w-0 flex-1">
          <VChart spec={spec} />
        </div>
        <div className="flex shrink-0 flex-col justify-center gap-2">
          {LEGEND_RECOMMENDATION_BUCKETS.map((bucket) => (
            <div key={bucket.key} className="flex items-center gap-2 text-xs text-slate-400">
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
    <div className="rounded-md border border-white/10">
      <div className="border-b border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-slate-500">
        Latest rating action
      </div>
      {latest ? (
        <RatingAction item={latest} />
      ) : (
        <p className="px-3 py-4 text-sm text-slate-500">No recent actions.</p>
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
    <div className="rounded-md border border-white/10">
      <div className="border-b border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-slate-500">
        Current month price changes
      </div>
      {total > 0 ? (
        <div className="px-3 py-2.5">
          <div className="mb-3 flex items-center justify-between text-xs">
            <span className="text-slate-500">{monthLabel}</span>
            <span className="font-medium text-slate-200">{total} updates</span>
          </div>
          <div className="flex h-3 overflow-hidden rounded-full bg-slate-700/80">
            <DistributionSegment
              count={counts.raises}
              total={total}
              color="#10b981"
              label="Raises"
            />
            <DistributionSegment
              count={counts.maintains}
              total={total}
              color="#f59e0b"
              label="Maintains"
            />
            <DistributionSegment
              count={counts.lowers}
              total={total}
              color="#ef4444"
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
            <DistributionRow label="Raises" count={counts.raises} color="#10b981" />
            <DistributionRow
              label="Maintains"
              count={counts.maintains}
              color="#f59e0b"
            />
            <DistributionRow label="Lowers" count={counts.lowers} color="#ef4444" />
            <DistributionRow label="Other" count={counts.other} color="#64748b" />
          </div>
        </div>
      ) : (
        <p className="px-3 py-4 text-sm text-slate-500">
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
      <span className="flex items-center gap-1.5 text-slate-500">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        {label}
      </span>
      <span className="font-medium text-slate-100">{count}</span>
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
    <div className="space-y-2 px-3 py-2.5 text-xs">
      <div className="flex justify-between gap-3">
        <span className="text-slate-500">Date</span>
        <span className="text-right text-slate-200">
          {formatDate(item.epochGradeDate)}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-slate-500">Analyst</span>
        <span className="truncate text-right font-medium text-slate-100">
          {textValue(item.firm)}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-slate-500">Rating action</span>
        <span className="text-right text-slate-200">
          {action}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-slate-500">Rating</span>
        <span className="text-right text-slate-200">
          {rating}
        </span>
      </div>
      <div className="flex justify-between gap-3">
        <span className="text-slate-500">Price Target</span>
        <span
          className={`text-right font-medium ${
            targetChange == null
              ? "text-slate-200"
              : targetChange >= 0
                ? "text-emerald-400"
                : "text-red-400"
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
    <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
      <div className="h-40 animate-pulse rounded-md border border-white/10 bg-muted/20" />
      <div className="h-48 animate-pulse rounded-md border border-white/10 bg-muted/20" />
      <div className="h-64 animate-pulse rounded-md border border-white/10 bg-muted/20" />
      <div className="h-64 animate-pulse rounded-md border border-white/10 bg-muted/20" />
    </div>
  );
}
