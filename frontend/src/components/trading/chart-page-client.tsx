"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useState } from "react";
import { TopNav } from "@/components/nav";
import Container from "@/components/container";
import TradingChart from "@/components/trading/TradingChart";
import ChartControls from "@/components/trading/chart-controls";
import { defaultDateRange } from "@/lib/chart-data";
import type { AnalysisChartResponse } from "@/lib/types";
export default function ChartPageClient() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const defaults = defaultDateRange();

  const symbol = searchParams.get("symbol") ?? "AAPL";
  const start = searchParams.get("start") ?? defaults.start;
  const end = searchParams.get("end") ?? defaults.end;
  const indicatorsParam = searchParams.get("indicators") ?? "sma:5,sma:20";

  const [chartData, setChartData] = useState<AnalysisChartResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  const onData = useCallback(
    (data: AnalysisChartResponse) => {
      setChartData(data);
      const params = new URLSearchParams({
        symbol: data.symbol,
        start: data.start,
        end: data.end,
      });
      const ind = searchParams.get("indicators");
      if (ind) params.set("indicators", ind);
      router.replace(`/chart?${params.toString()}`, { scroll: false });
    },
    [router, searchParams],
  );

  return (
    <>
      <TopNav title="Chart" />
      <Container className="space-y-4 py-6">
        <ChartControls
          initialSymbol={symbol}
          initialStart={start}
          initialEnd={end}
          initialIndicators={indicatorsParam}
          onData={onData}
          onError={setError}
        />
        {error && (
          <p className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}
        {chartData && (
          <div className="flex flex-wrap gap-2 text-sm">
            <span className="rounded-full bg-secondary px-3 py-1">
              {chartData.symbol} · {chartData.interval}
            </span>
            <span className="rounded-full bg-secondary px-3 py-1">
              source: {chartData.meta.source}
            </span>
            <span className="rounded-full bg-secondary px-3 py-1">
              bars: {chartData.meta.bar_count}
            </span>
            {chartData.meta.cached_through && (
              <span className="rounded-full bg-secondary px-3 py-1">
                cached through {chartData.meta.cached_through}
              </span>
            )}
          </div>
        )}
        {chartData ? (
          <TradingChart data={chartData} />
        ) : (
          <div className="flex h-[480px] items-center justify-center rounded-lg border border-dashed border-border text-muted-foreground">
            Enter a symbol and click Load chart. Ensure the API is running on
            port 8000.
          </div>
        )}
      </Container>
    </>
  );
}
