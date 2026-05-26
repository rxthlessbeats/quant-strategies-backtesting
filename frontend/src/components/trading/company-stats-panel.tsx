"use client";

import type { CompanyOverview } from "@/lib/types";

const STATS: Array<[keyof CompanyOverview, string, "compact" | "percent" | "raw"]> = [
  ["eps", "EPS", "raw"],
  ["profit_margin", "Profit Margin", "percent"],
  ["operating_margin_ttm", "Operating Margin", "percent"],
  ["return_on_equity_ttm", "Return on Equity", "percent"],
  ["revenue_ttm", "Revenue TTM", "compact"],
  ["gross_profit_ttm", "Gross Profit TTM", "compact"],
  ["ebitda", "EBITDA", "compact"],
  ["revenue_per_share_ttm", "Revenue / Share TTM", "raw"],
  ["quarterly_earnings_growth_yoy", "Earnings Growth YoY", "percent"],
  ["quarterly_revenue_growth_yoy", "Revenue Growth YoY", "percent"],
  ["shares_outstanding", "Shares Outstanding", "compact"],
];

interface CompanyStatsPanelProps {
  overview: CompanyOverview | null;
  loading: boolean;
}

export default function CompanyStatsPanel({
  overview,
  loading,
}: CompanyStatsPanelProps) {
  return (
    <div className="rounded-lg border border-border bg-black/40 px-3 py-3 backdrop-blur-md">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-medium text-slate-200">Fundamental Stats</h2>
        {overview?.latest_quarter && (
          <span className="text-xs text-slate-500">
            Latest quarter: {overview.latest_quarter}
          </span>
        )}
      </div>

      {loading ? (
        <p className="text-sm text-slate-500">Loading fundamentals...</p>
      ) : (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {STATS.map(([key, label, format]) => (
            <div key={key} className="rounded-md border border-white/10 p-3">
              <div className="text-xs text-slate-500">{label}</div>
              <div className="mt-1 text-sm font-medium text-slate-100">
                {formatStat(overview?.[key], format)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatStat(
  value: string | number | null | undefined,
  format: "compact" | "percent" | "raw",
): string {
  if (value == null || value === "" || value === "None") return "N/A";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);

  if (format === "compact") {
    return Intl.NumberFormat(undefined, {
      maximumFractionDigits: 2,
      notation: "compact",
    }).format(numeric);
  }

  if (format === "percent") {
    return `${(numeric * 100).toFixed(2)}%`;
  }

  return Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(
    numeric,
  );
}
