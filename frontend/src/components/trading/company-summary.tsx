"use client";

import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { formatDailyMarketAsOf } from "@/lib/market-timestamps";
import type { LatestQuote } from "@/lib/quote-from-bars";
import type { CompanyOverview } from "@/lib/types";
import { cn } from "@/lib/utils";

interface CompanySummaryProps {
  overview: CompanyOverview | null;
  overviewLoading: boolean;
  quote: LatestQuote | null;
  quoteLoading?: boolean;
}

function formatPrice(value: number): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function ChangeArrow({ up, down }: { up: boolean; down: boolean }) {
  if (up) {
    return (
      <ArrowUpRight className="ml-0.5 inline-block h-3 w-3 shrink-0" aria-hidden />
    );
  }
  if (down) {
    return (
      <ArrowDownRight
        className="ml-0.5 inline-block h-3 w-3 shrink-0"
        aria-hidden
      />
    );
  }
  return null;
}

function ChangeBlock({ quote }: { quote: LatestQuote }) {
  const positive = quote.changeAmount > 0;
  const negative = quote.changeAmount < 0;
  const neutral = !positive && !negative;

  return (
    <div
      className={cn(
        "flex h-full flex-col justify-center text-left text-sm font-medium leading-tight tabular-nums",
        positive && "text-emerald-500",
        negative && "text-red-500",
        neutral && "text-muted-foreground",
      )}
    >
      <span className="inline-flex items-center">
        {positive ? "+" : ""}
        {formatPrice(quote.changeAmount)}
        <ChangeArrow up={positive} down={negative} />
      </span>
      <span className="inline-flex items-center">
        {positive ? "+" : ""}
        {(quote.changePercent * 100).toFixed(2)}%
        <ChangeArrow up={positive} down={negative} />
      </span>
    </div>
  );
}

export default function CompanySummary({
  overview,
  overviewLoading,
  quote,
  quoteLoading = false,
}: CompanySummaryProps) {
  const showHeader =
    overviewLoading || overview !== null || quoteLoading || quote !== null;
  const quoteAsOf = formatDailyMarketAsOf(quote?.asOf);

  if (!showHeader) {
    return null;
  }

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-x-6 gap-y-1">
      {overviewLoading ? (
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
            <div className="h-9 w-28 animate-pulse rounded bg-muted" />
            <div className="h-6 w-48 animate-pulse rounded bg-muted" />
          </div>
        </div>
      ) : overview ? (
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
            <span className="text-4xl font-semibold leading-none tracking-tight">
              {overview.symbol}
            </span>
            <span className="text-2xl font-medium leading-none text-muted-foreground">
              {overview.name ?? "N/A"}
            </span>
          </div>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Loading company details…</p>
      )}

      <div className="shrink-0 justify-self-end text-right">
        {quoteLoading ? (
          <div className="flex items-stretch justify-end gap-3">
            <div className="h-9 w-28 animate-pulse rounded bg-muted" />
            <div className="flex flex-col justify-center gap-1">
              <div className="h-4 w-16 animate-pulse rounded bg-muted" />
              <div className="h-4 w-16 animate-pulse rounded bg-muted" />
            </div>
          </div>
        ) : quote ? (
          <div className="flex items-stretch justify-end gap-3">
            <div
              className={cn(
                "text-4xl font-semibold leading-none tabular-nums tracking-tight",
                quote.changeAmount > 0 && "text-emerald-500",
                quote.changeAmount < 0 && "text-red-500",
                quote.changeAmount === 0 && "text-foreground",
              )}
            >
              {formatPrice(quote.price)}
            </div>
            <ChangeBlock quote={quote} />
          </div>
        ) : null}
      </div>

      <div className="min-w-0 text-sm text-muted-foreground">
        {overviewLoading ? (
          <div className="h-5 w-64 animate-pulse rounded bg-muted" />
        ) : overview
          ? [overview.sector, overview.industry].filter(Boolean).join(" • ") ||
            "N/A"
          : null}
      </div>

      <div className="justify-self-end text-right text-sm text-muted-foreground">
        {quoteLoading ? (
          <div className="ml-auto h-5 w-40 animate-pulse rounded bg-muted" />
        ) : quoteAsOf ? (
          `As of ${quoteAsOf}`
        ) : null}
      </div>
    </div>
  );
}
