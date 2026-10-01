"use client";

import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { formatDailyMarketAsOf } from "@/lib/market-timestamps";
import type { LatestQuote } from "@/lib/quote-from-bars";
import type { CompanyOverview } from "@/lib/types";

interface CompanySummaryProps {
  symbol: string;
  overview: CompanyOverview | null;
  overviewLoading: boolean;
  quote: LatestQuote | null;
  quoteLoading?: boolean;
}

export default function CompanySummary({ symbol, overview, overviewLoading, quote, quoteLoading = false }: CompanySummaryProps) {
  const quoteAsOf = formatDailyMarketAsOf(quote?.asOf);
  const positive = quote && quote.changeAmount >= 0;
  return (
    <div className="company-summary">
      <div className="company-identity">
        <div className="workspace-context">Research workspace <span>/</span> {overview?.exchange || "US equities"}</div>
        <div className="company-title"><h1>{symbol}</h1><span>{overviewLoading ? "Loading company…" : overview?.name || "Company research"}</span></div>
        <p>{[overview?.sector, overview?.industry].filter(Boolean).join(" / ") || "Price action, fundamentals, and market perspective"}</p>
      </div>
      <div className="company-quote">
        {quoteLoading ? <div className="quote-skeleton" role="status" aria-label="Loading quote" /> : quote ? <>
          <div className="quote-price">{quote.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}<span>{overview?.currency || "USD"}</span></div>
          <div className={`quote-change ${positive ? "positive" : "negative"}`}>{positive ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}{positive ? "+" : ""}{quote.changeAmount.toFixed(2)} <span>({positive ? "+" : ""}{(quote.changePercent * 100).toFixed(2)}%)</span></div>
        </> : <p className="quote-unavailable">Quote unavailable</p>}
        <small>{quoteAsOf ? `Daily close · ${quoteAsOf}` : "Daily prices, not real time"}</small>
      </div>
    </div>
  );
}
