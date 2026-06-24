"use client";

import { VChart } from "@visactor/react-vchart";
import type { IBarChartSpec } from "@visactor/vchart";
import { useMemo, useState } from "react";
import type { CompanyOverview, MarketDataAreaResponse } from "@/lib/types";

interface CompanyStatsPanelProps {
  overview: CompanyOverview | null;
  data: MarketDataAreaResponse | null;
  statements: MarketDataAreaResponse | null;
  earnings: MarketDataAreaResponse | null;
  loading: boolean;
}

interface YahooValue {
  raw?: unknown;
  fmt?: unknown;
}

interface FinancialRow {
  label: string;
  value: string;
}

interface FinancialSection {
  title: string;
  rows: FinancialRow[];
}

export default function CompanyStatsPanel({
  overview,
  data,
  statements,
  earnings,
  loading,
}: CompanyStatsPanelProps) {
  const [earningsView, setEarningsView] = useState<"quarterly" | "yearly">(
    "quarterly",
  );
  const sections = useMemo(
    () => buildFinancialSections(overview, data),
    [overview, data],
  );
  const chartRows = useMemo(
    () => buildEarningsTrendChartRows(earnings, data, overview, earningsView),
    [earnings, data, overview, earningsView],
  );
  const revenueEarningsRows = useMemo(
    () => buildRevenueEarningsChartRows(statements, earningsView),
    [statements, earningsView],
  );
  const growthSummary = useMemo(
    () =>
      buildEarningsGrowthSummary(
        earnings,
        statements,
        data,
        overview,
        earningsView,
      ),
    [earnings, statements, data, overview, earningsView],
  );
  const cashDebtRows = useMemo(
    () => buildCashDebtChartRows(statements, data, earningsView),
    [statements, data, earningsView],
  );

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="rounded-lg border border-border bg-black/40 px-3 py-3 backdrop-blur-md">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium text-slate-200">Financials</h2>
          {overview?.latest_quarter && (
            <span className="text-xs text-slate-500">
              Latest quarter: {overview.latest_quarter}
            </span>
          )}
        </div>

        {loading ? (
          <p className="text-sm text-slate-500">Loading fundamentals...</p>
        ) : (
          <div className="space-y-4">
            {sections.map((section) => (
              <div key={section.title}>
                <div className="mb-1 text-xs font-medium text-slate-300">
                  {section.title}
                </div>
                <div className="divide-y divide-white/10 border-t border-white/10">
                  {section.rows.map((row) => (
                    <div
                      key={row.label}
                      className="flex items-start justify-between gap-3 py-2 text-xs"
                    >
                      <span className="text-slate-500">{row.label}</span>
                      <span className="text-right font-medium text-slate-100">
                        {row.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="min-w-0 overflow-hidden rounded-lg border border-border bg-black/40 px-3 py-3 backdrop-blur-md">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-medium text-slate-200">Earnings Trends</h2>
          <div className="rounded-md border border-white/10 bg-black/30 p-0.5 text-xs">
            {(["quarterly", "yearly"] as const).map((view) => (
              <button
                key={view}
                type="button"
                onClick={() => {
                  if (earningsView !== view) {
                    setEarningsView(view);
                  }
                }}
                className={`rounded px-2 py-1 capitalize ${
                  earningsView === view
                    ? "bg-white/10 text-slate-100"
                    : "text-slate-500 hover:text-slate-200"
                }`}
              >
                {view}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <p className="text-sm text-slate-500">Loading earnings...</p>
        ) : chartRows.length ? (
          <div className="space-y-4">
            <div>
              <p className="mb-2 text-xs font-medium text-slate-300">
                EPS estimate vs actual
              </p>
              <EarningsTrendChart rows={chartRows} />
            </div>
            {revenueEarningsRows.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium text-slate-300">
                  Revenue vs Earnings
                </p>
                <RevenueEarningsChart rows={revenueEarningsRows} />
              </div>
            )}
            {cashDebtRows.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium text-slate-300">
                  Cash vs Debt
                </p>
                <CashDebtChart rows={cashDebtRows} />
              </div>
            )}
            {growthSummary && (
              <EarningsGrowthTable summary={growthSummary} view={earningsView} />
            )}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No earnings trend data available.</p>
        )}
      </div>
    </div>
  );
}

interface EarningsTrendChartRow {
  label: string;
  value: number;
  type: "EPS Estimate" | "Actual EPS";
}

interface RevenueEarningsChartRow {
  label: string;
  value: number;
  type: "Revenue" | "Earnings";
}

interface CashDebtChartRow {
  label: string;
  value: number;
  type: "Cash" | "Debt";
}

interface GrowthRates {
  qoq: string;
  yoy: string;
}

interface EarningsGrowthSummary {
  periodLabel: string;
  eps: GrowthRates;
  revenue: GrowthRates;
  earnings: GrowthRates;
}

function EarningsGrowthTable({
  summary,
  view,
}: {
  summary: EarningsGrowthSummary;
  view: "quarterly" | "yearly";
}) {
  const rows: { metric: string; rates: GrowthRates }[] = [
    { metric: "EPS", rates: summary.eps },
    { metric: "Revenue", rates: summary.revenue },
    { metric: "Earnings", rates: summary.earnings },
  ];

  const gridClass =
    view === "quarterly"
      ? "grid-cols-[1fr_4.5rem_4.5rem]"
      : "grid-cols-[1fr_4.5rem]";

  return (
    <div className="rounded-md border border-white/10 bg-white/[0.02] px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="text-xs font-medium text-slate-300">Growth</p>
        <span className="text-xs text-slate-500">{summary.periodLabel}</span>
      </div>
      <div className={`grid ${gridClass} gap-x-3 gap-y-1 text-xs`}>
        <div className="text-slate-500" />
        {view === "quarterly" && (
          <div className="text-right text-slate-500">QoQ</div>
        )}
        <div className="text-right text-slate-500">YoY</div>
        {rows.map((row) => (
          <div key={row.metric} className="contents">
            <span className="text-slate-400">{row.metric}</span>
            {view === "quarterly" && <GrowthCell value={row.rates.qoq} />}
            <GrowthCell value={row.rates.yoy} />
          </div>
        ))}
      </div>
    </div>
  );
}

function GrowthCell({ value }: { value: string }) {
  const positive = value.startsWith("+");
  const negative = value.startsWith("-") && value !== "—";
  return (
    <span
      className={`text-right font-medium ${
        positive
          ? "text-emerald-400"
          : negative
            ? "text-rose-400"
            : "text-slate-500"
      }`}
    >
      {value}
    </span>
  );
}

function EarningsTrendChart({ rows }: { rows: EarningsTrendChartRow[] }) {
  const spec = useMemo<IBarChartSpec>(() => ({
    type: "bar",
    data: [
      {
        id: "earningsTrendData",
        values: rows,
      },
    ],
    xField: ["label", "type"],
    yField: "value",
    seriesField: "type",
    stack: false,
    barGapInGroup: "25%",
    height: 220,
    padding: [12, 12, 36, 4],
    color: ["#a78bfa", "#38bdf8"],
    legends: {
      visible: true,
      orient: "bottom",
      position: "middle",
      item: {
        label: {
          style: {
            fill: "#94a3b8",
          },
        },
      },
    },
    tooltip: {
      trigger: ["click", "hover"],
    },
    axes: [
      {
        orient: "left",
        title: {
          visible: true,
          text: "EPS",
          style: {
            fill: "#94a3b8",
          },
        },
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
      style: {
        cornerRadius: [4, 4, 0, 0],
      },
    },
  }), [rows]);

  return (
    <div className="h-60">
      <VChart spec={spec} />
    </div>
  );
}

function CashDebtChart({ rows }: { rows: CashDebtChartRow[] }) {
  const spec = useMemo<IBarChartSpec>(() => ({
    type: "bar",
    data: [
      {
        id: "cashDebtData",
        values: rows,
      },
    ],
    xField: ["label", "type"],
    yField: "value",
    seriesField: "type",
    stack: false,
    barGapInGroup: "25%",
    height: 220,
    padding: [12, 12, 36, 4],
    color: ["#22d3ee", "#f87171"],
    legends: {
      visible: true,
      orient: "bottom",
      position: "middle",
      item: {
        label: {
          style: {
            fill: "#94a3b8",
          },
        },
      },
    },
    tooltip: {
      trigger: ["click", "hover"],
    },
    axes: [
      {
        orient: "left",
        title: {
          visible: true,
          text: "USD (B)",
          style: {
            fill: "#94a3b8",
          },
        },
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
      style: {
        cornerRadius: [4, 4, 0, 0],
      },
    },
  }), [rows]);

  return (
    <div className="h-60">
      <VChart spec={spec} />
    </div>
  );
}

function RevenueEarningsChart({ rows }: { rows: RevenueEarningsChartRow[] }) {
  const spec = useMemo<IBarChartSpec>(() => ({
    type: "bar",
    data: [
      {
        id: "revenueEarningsData",
        values: rows,
      },
    ],
    xField: ["label", "type"],
    yField: "value",
    seriesField: "type",
    stack: false,
    barGapInGroup: "25%",
    height: 220,
    padding: [12, 12, 36, 4],
    color: ["#10b981", "#f59e0b"],
    legends: {
      visible: true,
      orient: "bottom",
      position: "middle",
      item: {
        label: {
          style: {
            fill: "#94a3b8",
          },
        },
      },
    },
    tooltip: {
      trigger: ["click", "hover"],
    },
    axes: [
      {
        orient: "left",
        title: {
          visible: true,
          text: "USD (B)",
          style: {
            fill: "#94a3b8",
          },
        },
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
      style: {
        cornerRadius: [4, 4, 0, 0],
      },
    },
  }), [rows]);

  return (
    <div className="h-60">
      <VChart spec={spec} />
    </div>
  );
}

function buildEarningsGrowthSummary(
  earnings: MarketDataAreaResponse | null,
  statements: MarketDataAreaResponse | null,
  statistics: MarketDataAreaResponse | null,
  overview: CompanyOverview | null,
  view: "quarterly" | "yearly",
): EarningsGrowthSummary | null {
  if (view === "yearly") {
    return buildYearlyEarningsGrowthSummary(
      earnings,
      statements,
      statistics,
      overview,
    );
  }
  return buildQuarterlyEarningsGrowthSummary(
    earnings,
    statements,
    statistics,
    overview,
  );
}

function buildQuarterlyEarningsGrowthSummary(
  earnings: MarketDataAreaResponse | null,
  statements: MarketDataAreaResponse | null,
  statistics: MarketDataAreaResponse | null,
  overview: CompanyOverview | null,
): EarningsGrowthSummary | null {
  const statementPayload = modulePayload(statements, "incomeStatementHistoryQuarterly");
  const history = isRecord(statementPayload)
    ? statementPayload.incomeStatementHistory
    : undefined;

  const financials = Array.isArray(history)
    ? history
        .filter(isRecord)
        .slice(0, 8)
        .reverse()
        .map((item) => ({
          label: quarterLabel(item.endDate),
          revenue: rawNumber(item.totalRevenue),
          earnings: rawNumber(item.netIncome),
        }))
    : [];

  const historyPayload = modulePayload(earnings, "earningsHistory");
  const epsHistory = isRecord(historyPayload) ? historyPayload.history : undefined;
  const epsSeries = Array.isArray(epsHistory)
    ? epsHistory
        .filter(isRecord)
        .map((item) => ({
          label: quarterLabel(item.quarter),
          eps: rawNumber(item.epsActual),
        }))
        .filter((item) => item.eps != null)
    : [];

  const latestFinancial = financials[financials.length - 1];
  const priorFinancial = financials[financials.length - 2];
  const latestEps = epsSeries[epsSeries.length - 1];
  const priorEps = epsSeries[epsSeries.length - 2];

  const trendItem = currentQuarterTrendItem(earnings);
  const earningsEstimate = isRecord(trendItem?.earningsEstimate)
    ? trendItem.earningsEstimate
    : null;
  const revenueEstimate = isRecord(trendItem?.revenueEstimate)
    ? trendItem.revenueEstimate
    : null;

  const financial = modulePayload(statistics, "financialData");

  const periodLabel =
    latestFinancial?.label ?? latestEps?.label ?? "-";
  if (periodLabel === "-" && !latestFinancial && !latestEps) return null;

  return {
    periodLabel,
    eps: {
      qoq: formatGrowthPercent(latestEps?.eps ?? null, priorEps?.eps ?? null),
      yoy: growthYoY(
        latestEps?.eps ?? null,
        rawNumber(earningsEstimate?.yearAgoEps),
        earningsEstimate?.growth,
      ),
    },
    revenue: {
      qoq: formatGrowthPercent(
        latestFinancial?.revenue ?? null,
        priorFinancial?.revenue ?? null,
      ),
      yoy: growthYoY(
        latestFinancial?.revenue ?? null,
        rawNumber(revenueEstimate?.yearAgoRevenue),
        revenueEstimate?.growth,
      ),
    },
    earnings: {
      qoq: formatGrowthPercent(
        latestFinancial?.earnings ?? null,
        priorFinancial?.earnings ?? null,
      ),
      yoy: formatGrowthRate(
        financial?.earningsGrowth ?? overview?.quarterly_earnings_growth_yoy,
      ),
    },
  };
}

function buildYearlyEarningsGrowthSummary(
  earnings: MarketDataAreaResponse | null,
  _statements: MarketDataAreaResponse | null,
  statistics: MarketDataAreaResponse | null,
  overview: CompanyOverview | null,
): EarningsGrowthSummary | null {
  const shares =
    sharesOutstanding(statistics) ?? rawNumber(overview?.shares_outstanding);
  const earningsPayload = modulePayload(earnings, "earnings");
  const fc = isRecord(earningsPayload) ? earningsPayload.financialsChart : null;
  if (!isRecord(fc) || !Array.isArray(fc.yearly)) return null;

  const series = fc.yearly
    .filter(isRecord)
    .map((item) => {
      const label = String(item.date ?? "");
      const netIncome = rawNumber(item.earnings);
      return {
        label,
        revenue: rawNumber(item.revenue),
        earnings: netIncome,
        eps:
          netIncome != null && shares != null && shares > 0
            ? Number((netIncome / shares).toFixed(4))
            : null,
      };
    })
    .filter((item) => item.label && item.label !== "undefined");

  if (!series.length) return null;

  const latest = series[series.length - 1];
  const prior = series.length >= 2 ? series[series.length - 2] : null;

  return {
    periodLabel: latest.label,
    eps: {
      qoq: "—",
      yoy: formatGrowthPercent(latest.eps, prior?.eps ?? null),
    },
    revenue: {
      qoq: "—",
      yoy: formatGrowthPercent(latest.revenue, prior?.revenue ?? null),
    },
    earnings: {
      qoq: "—",
      yoy: formatGrowthPercent(latest.earnings, prior?.earnings ?? null),
    },
  };
}

function currentQuarterTrendItem(
  earnings: MarketDataAreaResponse | null,
): Record<string, unknown> | null {
  const payload = modulePayload(earnings, "earningsTrend");
  const trend = isRecord(payload) ? payload.trend : undefined;
  if (!Array.isArray(trend)) return null;
  const item = trend.find(
    (entry) => isRecord(entry) && textValue(entry.period) === "0q",
  );
  return isRecord(item) ? item : null;
}

function growthYoY(
  current: number | null,
  yearAgo: number | null,
  fallbackRate: unknown,
): string {
  const computed = formatGrowthPercent(current, yearAgo);
  return computed !== "—" ? computed : formatGrowthRate(fallbackRate);
}

function formatGrowthPercent(
  current: number | null,
  prior: number | null,
): string {
  if (current == null || prior == null || prior === 0) return "—";
  const change = ((current - prior) / Math.abs(prior)) * 100;
  const sign = change > 0 ? "+" : "";
  return `${sign}${change.toFixed(1)}%`;
}

function formatGrowthRate(value: unknown): string {
  const raw = rawNumber(value);
  if (raw == null) return "—";
  const pct = Math.abs(raw) <= 1.5 ? raw * 100 : raw;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

function buildEarningsTrendChartRows(
  data: MarketDataAreaResponse | null,
  statistics: MarketDataAreaResponse | null,
  overview: CompanyOverview | null,
  view: "quarterly" | "yearly",
): EarningsTrendChartRow[] {
  if (view === "yearly") {
    return buildYearlyEarningsTrendChartRows(data, statistics, overview);
  }

  const historyPayload = modulePayload(data, "earningsHistory");
  const history = isRecord(historyPayload) ? historyPayload.history : undefined;
  const pastRows: EarningsTrendChartRow[] = [];
  if (Array.isArray(history)) {
    history
      .filter(isRecord)
      .slice(-3)
      .forEach((item) => {
        const label = quarterLabel(item.quarter);
        const estimate = rawNumber(item.epsEstimate);
        const actual = rawNumber(item.epsActual);
        if (estimate != null) {
          pastRows.push({ label, value: estimate, type: "EPS Estimate" });
        }
        if (actual != null) {
          pastRows.push({ label, value: actual, type: "Actual EPS" });
        }
      });
  }

  const trendPayload = modulePayload(data, "earningsTrend");
  const trend = isRecord(trendPayload) ? trendPayload.trend : undefined;
  const futureRows: EarningsTrendChartRow[] = [];
  if (Array.isArray(trend)) {
    trend
      .filter(isRecord)
      .filter((item) => textValue(item.period).endsWith("q"))
      .slice(0, 2)
      .forEach((item) => {
        const earningsEstimate = isRecord(item.earningsEstimate)
          ? item.earningsEstimate
          : null;
        const value = rawNumber(earningsEstimate?.avg);
        if (value != null) {
          futureRows.push({
            label: quarterLabel(item.endDate),
            value,
            type: "EPS Estimate",
          });
        }
      });
  }

  return [...pastRows, ...futureRows];
}

function buildYearlyEarningsTrendChartRows(
  data: MarketDataAreaResponse | null,
  statistics: MarketDataAreaResponse | null,
  overview: CompanyOverview | null,
): EarningsTrendChartRow[] {
  const rows: EarningsTrendChartRow[] = [];
  const shares =
    sharesOutstanding(statistics) ?? rawNumber(overview?.shares_outstanding);

  // Primary: annual net income ÷ shares outstanding from financialsChart.yearly
  const earningsPayload = modulePayload(data, "earnings");
  if (isRecord(earningsPayload) && shares != null && shares > 0) {
    const fc = earningsPayload.financialsChart;
    if (isRecord(fc) && Array.isArray(fc.yearly)) {
      fc.yearly
        .filter(isRecord)
        .slice(-4)
        .forEach((item) => {
          const netIncome = rawNumber(item.earnings);
          const year = String(item.date ?? "");
          if (netIncome == null || !year || year === "undefined") return;
          rows.push({
            label: year,
            value: Number((netIncome / shares).toFixed(2)),
            type: "Actual EPS",
          });
        });
    }
  }

  // Fallback: derive from earningsTrend yearAgoEps when primary has no data
  if (rows.length === 0) {
    const trendPayload = modulePayload(data, "earningsTrend");
    if (isRecord(trendPayload) && Array.isArray(trendPayload.trend)) {
      trendPayload.trend
        .filter(isRecord)
        .filter((item) => textValue(item.period).endsWith("y"))
        .forEach((item) => {
          const est = isRecord(item.earningsEstimate) ? item.earningsEstimate : null;
          const yearAgo = rawNumber(est?.yearAgoEps);
          const endDate = textValue(item.endDate);
          const priorYear = endDate !== "-"
            ? String(new Date(endDate).getUTCFullYear() - 1)
            : null;
          if (yearAgo != null && priorYear && !rows.some((r) => r.label === priorYear)) {
            rows.push({ label: priorYear, value: yearAgo, type: "Actual EPS" });
          }
        });
    }
  }

  return rows;
}

function buildCashDebtChartRows(
  statements: MarketDataAreaResponse | null,
  statistics: MarketDataAreaResponse | null,
  view: "quarterly" | "yearly",
): CashDebtChartRow[] {
  const periodFilter = view === "yearly" ? "12M" : "3M";

  let cashPoints = fundamentalsSeriesPoints(
    statements,
    view === "yearly"
      ? "annualCashAndCashEquivalents"
      : "quarterlyCashAndCashEquivalents",
    periodFilter,
  );
  let debtPoints = fundamentalsSeriesPoints(
    statements,
    view === "yearly" ? "annualTotalDebt" : "quarterlyTotalDebt",
    periodFilter,
  );

  if (view === "quarterly") {
    const cashAltPoints = fundamentalsSeriesPoints(
      statements,
      "quarterlyCashCashEquivalentsAndShortTermInvestments",
      periodFilter,
    );
    cashPoints = mergeFundamentalsPoints(cashPoints, cashAltPoints);
  }

  // Yearly: if annual series are not cached yet, use quarterly history instead.
  if (view === "yearly" && (cashPoints.length === 0 || debtPoints.length === 0)) {
    const qCash = fundamentalsSeriesPoints(
      statements,
      "quarterlyCashAndCashEquivalents",
      "3M",
    );
    const qDebt = fundamentalsSeriesPoints(
      statements,
      "quarterlyTotalDebt",
      "3M",
    );
    if (cashPoints.length === 0) cashPoints = qCash;
    if (debtPoints.length === 0) debtPoints = qDebt;
  }

  const rows = buildCashDebtRowsFromPoints(cashPoints, debtPoints, view);
  if (rows.length) return rows;

  const financial = modulePayload(statistics, "financialData");
  const cash = rawNumber(financial?.totalCash);
  const debt = rawNumber(financial?.totalDebt);
  if (cash == null && debt == null) return [];

  const label = view === "yearly" ? "Latest FY" : "Latest Q";
  const fallback: CashDebtChartRow[] = [];
  const cashB = billionsValue(cash);
  const debtB = billionsValue(debt);
  if (cashB != null) fallback.push({ label, value: cashB, type: "Cash" });
  if (debtB != null) fallback.push({ label, value: debtB, type: "Debt" });
  return fallback;
}

function mergeFundamentalsPoints(
  primary: FundamentalsPoint[],
  secondary: FundamentalsPoint[],
): FundamentalsPoint[] {
  const byDate = new Map(primary.map((point) => [point.asOfDate, point]));
  secondary.forEach((point) => {
    if (!byDate.has(point.asOfDate)) {
      byDate.set(point.asOfDate, point);
    }
  });
  return [...byDate.values()].sort((a, b) =>
    a.asOfDate.localeCompare(b.asOfDate),
  );
}

function buildCashDebtRowsFromPoints(
  cashPoints: FundamentalsPoint[],
  debtPoints: FundamentalsPoint[],
  view: "quarterly" | "yearly",
): CashDebtChartRow[] {
  const byDate = new Map<
    string,
    { label: string; cash: number | null; debt: number | null }
  >();

  const mergePoints = (
    points: FundamentalsPoint[],
    field: "cash" | "debt",
  ) => {
    points.forEach((point) => {
      const existing = byDate.get(point.asOfDate) ?? {
        label: fundamentalsPeriodLabel(point.asOfDate, point.periodType, view),
        cash: null,
        debt: null,
      };
      existing[field] = point.value;
      byDate.set(point.asOfDate, existing);
    });
  };

  mergePoints(cashPoints, "cash");
  mergePoints(debtPoints, "debt");

  return [...byDate.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-4)
    .flatMap(([, item]) => {
      const rows: CashDebtChartRow[] = [];
      const cash = billionsValue(item.cash);
      const debt = billionsValue(item.debt);
      if (cash != null) rows.push({ label: item.label, value: cash, type: "Cash" });
      if (debt != null) rows.push({ label: item.label, value: debt, type: "Debt" });
      return rows;
    })
    .filter((row): row is CashDebtChartRow => row.value != null);
}

interface FundamentalsPoint {
  asOfDate: string;
  periodType: string;
  value: number;
}

function fundamentalsSeriesPoints(
  statements: MarketDataAreaResponse | null,
  type: string,
  periodFilter?: "3M" | "12M",
): FundamentalsPoint[] {
  const payload = modulePayload(statements, "fundamentalsTimeSeriesQuarterly");
  const timeseries = isRecord(payload?.timeseries) ? payload.timeseries : null;
  const result = timeseries && Array.isArray(timeseries.result)
    ? timeseries.result
    : undefined;
  if (!Array.isArray(result)) return [];

  const series = result.find(
    (item) => isRecord(item) && Array.isArray(item[type]),
  );
  if (!isRecord(series) || !Array.isArray(series[type])) return [];

  return series[type]
    .filter(isRecord)
    .map((item) => {
      const asOfDate = textValue(item.asOfDate);
      const periodType = textValue(item.periodType);
      const value = rawNumber(item.reportedValue);
      if (asOfDate === "-" || value == null) return null;
      if (periodFilter && periodType !== periodFilter) return null;
      return {
        asOfDate,
        periodType,
        value,
      };
    })
    .filter((item): item is FundamentalsPoint => item != null);
}

function fundamentalsPeriodLabel(
  asOfDate: string,
  periodType: string,
  view: "quarterly" | "yearly",
): string {
  const date = new Date(`${asOfDate}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return asOfDate;
  if (view === "yearly" || periodType === "12M") {
    return `FY ${date.getUTCFullYear()}`;
  }
  const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
  return `Q${quarter} ${date.getUTCFullYear()}`;
}

function buildRevenueEarningsChartRows(
  data: MarketDataAreaResponse | null,
  view: "quarterly" | "yearly",
): RevenueEarningsChartRow[] {
  if (view === "yearly") {
    const annualPayload = modulePayload(data, "incomeStatementHistory");
    const annualHistory = isRecord(annualPayload)
      ? annualPayload.incomeStatementHistory
      : undefined;
    const items = Array.isArray(annualHistory)
      ? annualHistory.filter(isRecord)
      : [];
    return buildYearlyRevenueEarningsRows(items);
  }

  const payload = modulePayload(data, "incomeStatementHistoryQuarterly");
  const history = isRecord(payload) ? payload.incomeStatementHistory : undefined;
  if (!Array.isArray(history)) return [];

  return history
    .filter(isRecord)
    .slice(0, 4)
    .reverse()
    .flatMap((item) => {
      const label = quarterLabel(item.endDate);
      return [
        {
          label,
          value: billionsValue(item.totalRevenue),
          type: "Revenue" as const,
        },
        {
          label,
          value: billionsValue(item.netIncome),
          type: "Earnings" as const,
        },
      ];
    })
    .filter((row): row is RevenueEarningsChartRow => row.value != null);
}

function buildYearlyRevenueEarningsRows(
  items: Record<string, unknown>[],
): RevenueEarningsChartRow[] {
  return items
    .slice(0, 4)
    .reverse()
    .map((item) => ({
      label: fiscalYearLabel(item.endDate),
      revenue: billionsValue(item.totalRevenue),
      earnings: billionsValue(item.netIncome),
    }))
    .filter((item) => item.label !== "-")
    .slice(-4)
    .flatMap((item) => [
      {
        label: item.label,
        value: item.revenue,
        type: "Revenue" as const,
      },
      {
        label: item.label,
        value: item.earnings,
        type: "Earnings" as const,
      },
    ])
    .filter((row): row is RevenueEarningsChartRow => row.value != null);
}

function quarterLabel(value: unknown): string {
  const text = textValue(value);
  if (text === "-") return text;
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return text;
  const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
  return `Q${quarter} ${date.getUTCFullYear()}`;
}

function fiscalYearLabel(value: unknown): string {
  const text = textValue(value);
  if (text === "-") return text;
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return text;
  return String(date.getUTCFullYear());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function textValue(value: unknown): string {
  if (value == null || value === "" || value === "None") return "-";
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
    if (yahooValue.raw != null) return textValue(yahooValue.raw);
  }
  return String(value);
}

function billionsValue(value: unknown): number | null {
  const raw = rawNumber(value);
  return raw == null ? null : Number((raw / 1_000_000_000).toFixed(2));
}

function buildFinancialSections(
  overview: CompanyOverview | null,
  data: MarketDataAreaResponse | null,
): FinancialSection[] {
  const stats = modulePayload(data, "defaultKeyStatistics");
  const financial = modulePayload(data, "financialData");

  return [
    {
      title: "Profitability",
      rows: [
        {
          label: "Profit Margin",
          value: percentValue(financial?.profitMargins ?? overview?.profit_margin),
        },
        {
          label: "Operating Margin (ttm)",
          value: percentValue(financial?.operatingMargins ?? overview?.operating_margin_ttm),
        },
        {
          label: "Gross Margin (ttm)",
          value: percentValue(financial?.grossMargins),
        },
        {
          label: "EBITDA Margin (ttm)",
          value: percentValue(financial?.ebitdaMargins),
        },
      ],
    },
    {
      title: "Management Effectiveness",
      rows: [
        {
          label: "Return on Assets (ttm)",
          value: percentValue(financial?.returnOnAssets ?? overview?.return_on_assets_ttm),
        },
        {
          label: "Return on Equity (ttm)",
          value: percentValue(financial?.returnOnEquity ?? overview?.return_on_equity_ttm),
        },
      ],
    },
    {
      title: "Income Statement",
      rows: [
        {
          label: "Revenue (ttm)",
          value: moneyValue(financial?.totalRevenue ?? overview?.revenue_ttm),
        },
        {
          label: "Revenue Per Share (ttm)",
          value: numberValue(financial?.revenuePerShare ?? overview?.revenue_per_share_ttm),
        },
        {
          label: "Quarterly Revenue Growth (yoy)",
          value: percentValue(financial?.revenueGrowth ?? overview?.quarterly_revenue_growth_yoy),
        },
        {
          label: "Gross Profit (ttm)",
          value: moneyValue(financial?.grossProfits ?? overview?.gross_profit_ttm),
        },
        {
          label: "EBITDA",
          value: moneyValue(financial?.ebitda ?? overview?.ebitda),
        },
        {
          label: "Net Income Avi to Common (ttm)",
          value: moneyValue(financial?.netIncomeToCommon),
        },
        {
          label: "Diluted EPS (ttm)",
          value: numberValue(stats?.trailingEps ?? overview?.diluted_eps_ttm ?? overview?.eps),
        },
        {
          label: "Quarterly Earnings Growth (yoy)",
          value: percentValue(financial?.earningsGrowth ?? overview?.quarterly_earnings_growth_yoy),
        },
      ],
    },
    {
      title: "Balance Sheet",
      rows: [
        {
          label: "Total Cash (mrq)",
          value: moneyValue(financial?.totalCash),
        },
        {
          label: "Total Cash Per Share (mrq)",
          value: numberValue(financial?.totalCashPerShare),
        },
        {
          label: "Total Debt (mrq)",
          value: moneyValue(financial?.totalDebt),
        },
        {
          label: "Total Debt/Equity (mrq)",
          value: numberValue(financial?.debtToEquity),
        },
        {
          label: "Current Ratio (mrq)",
          value: numberValue(financial?.currentRatio),
        },
        {
          label: "Book Value Per Share (mrq)",
          value: numberValue(stats?.bookValue ?? overview?.book_value),
        },
      ],
    },
    {
      title: "Cash Flow Statement",
      rows: [
        {
          label: "Operating Cash Flow (ttm)",
          value: moneyValue(financial?.operatingCashflow),
        },
        {
          label: "Levered Free Cash Flow (ttm)",
          value: moneyValue(financial?.freeCashflow),
        },
      ],
    },
  ];
}

function modulePayload(data: MarketDataAreaResponse | null, module: string) {
  return data?.modules.find((item) => item.module === module)?.payload ?? null;
}

function sharesOutstanding(data: MarketDataAreaResponse | null): number | null {
  const stats = modulePayload(data, "defaultKeyStatistics");
  return rawNumber(stats?.sharesOutstanding);
}

function rawNumber(value: unknown): number | null {
  if (value == null || value === "" || value === "None") return null;
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
  if (value == null || value === "" || value === "None") return fallback;
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
    if (yahooValue.raw != null) return formattedValue(yahooValue.raw, fallback);
  }
  return String(value);
}

function compactNumber(value: number): string {
  return Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
    notation: "compact",
  }).format(value);
}

function moneyValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
  }
  const raw = rawNumber(value);
  return raw == null ? formattedValue(value) : compactNumber(raw);
}

function numberValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
  }
  const raw = rawNumber(value);
  if (raw == null) return formattedValue(value);
  return Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
  }).format(raw);
}

function percentValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "object") {
    const yahooValue = value as YahooValue;
    if (yahooValue.fmt != null) return String(yahooValue.fmt);
  }
  const raw = rawNumber(value);
  if (raw == null) return formattedValue(value);
  return Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 2,
  }).format(raw);
}
