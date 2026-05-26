"use client";

import { Check, ChevronDown, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { fetchPerformanceBenchmarkOptions } from "@/lib/api";
import type {
  PerformanceBenchmarkGroup,
  PerformanceComparisonResponse,
} from "@/lib/types";
import { cn } from "@/lib/utils";

interface PerformanceComparisonPanelProps {
  data: PerformanceComparisonResponse | null;
  loading: boolean;
  error: string | null;
  benchmark: string;
  onBenchmarkChange: (benchmark: string) => void;
}

const LOADING_ROWS = ["1W", "1M", "1Q", "6M", "YTD", "1Y", "3Y", "5Y"];

export default function PerformanceComparisonPanel({
  data,
  loading,
  error,
  benchmark,
  onBenchmarkChange,
}: PerformanceComparisonPanelProps) {
  const [searchInput, setSearchInput] = useState("");
  const [benchmarkGroups, setBenchmarkGroups] = useState<
    PerformanceBenchmarkGroup[]
  >([]);
  const benchmarkOptions = useMemo(
    () => benchmarkGroups.flatMap((group) => group.options),
    [benchmarkGroups],
  );
  const benchmarkUpper = benchmark.toUpperCase();
  const selectedBenchmark = benchmarkOptions.find(
    (option) => option.symbol === benchmarkUpper,
  );

  useEffect(() => {
    let cancelled = false;
    fetchPerformanceBenchmarkOptions()
      .then((response) => {
        if (!cancelled) setBenchmarkGroups(response.groups);
      })
      .catch(() => {
        if (!cancelled) setBenchmarkGroups([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const chooseBenchmark = (value: string) => {
    onBenchmarkChange(value.trim().toUpperCase());
    setSearchInput("");
  };

  const handleCustomSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextBenchmark = searchInput.trim().toUpperCase();
    if (!nextBenchmark) return;
    chooseBenchmark(nextBenchmark);
  };

  return (
    <div className="rounded-lg border border-border bg-black/40 px-3 py-3 backdrop-blur-md">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-medium text-slate-200">
          Performance Comparison
        </h2>
        <div className="flex flex-wrap items-center justify-end gap-3">
          {data?.as_of && !loading && (
            <span className="text-xs text-slate-500">As of {data.as_of}</span>
          )}
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span>Benchmark: ETFs or custom</span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className={cn(
                    "h-8 gap-1.5 border border-white/10 bg-black/30 px-2",
                    "text-xs text-slate-200 hover:bg-white/10 hover:text-slate-100",
                  )}
                >
                  {selectedBenchmark?.symbol ?? "Select ETF"}
                  <ChevronDown className="h-3.5 w-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="max-h-96 w-72 overflow-y-auto">
                {benchmarkGroups.map((group, groupIndex) => (
                  <div key={group.category}>
                    {groupIndex > 0 && <DropdownMenuSeparator />}
                    <DropdownMenuLabel className="text-xs text-muted-foreground">
                      {group.category}
                    </DropdownMenuLabel>
                    {group.options.map((option) => (
                      <BenchmarkItem
                        key={option.symbol}
                        value={option.symbol}
                        label={option.symbol}
                        description={option.description}
                        selected={option.symbol === benchmarkUpper}
                        onSelect={chooseBenchmark}
                      />
                    ))}
                  </div>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <form
            onSubmit={handleCustomSubmit}
            className="flex items-center gap-2 text-xs text-slate-500"
          >
            <span>or Custom</span>
            <div className="relative">
              <input
                type="text"
                value={searchInput}
                onChange={(event) =>
                  setSearchInput(event.target.value.toUpperCase())
                }
                placeholder="Any symbol"
                className={cn(
                  "h-8 w-32 rounded-md border border-white/10 bg-black/30",
                  "py-1 pl-2 pr-8 text-xs uppercase text-slate-200 outline-none",
                  "placeholder:normal-case placeholder:text-muted-foreground",
                  "hover:bg-white/10 focus:border-white/30",
                )}
              />
              <button
                type="submit"
                aria-label="Set custom benchmark"
                className={cn(
                  "absolute right-1 top-1/2 flex h-6 w-6 -translate-y-1/2",
                  "items-center justify-center rounded text-muted-foreground",
                  "hover:bg-white/10 hover:text-slate-100",
                )}
              >
                <Search className="h-3.5 w-3.5" />
              </button>
            </div>
          </form>
        </div>
      </div>

      {error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : loading ? (
        <LoadingTable />
      ) : data ? (
        <div className="overflow-x-auto rounded-md border border-white/10">
          <div className="min-w-[760px]">
            <div className="grid grid-cols-9 border-b border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-slate-500">
              <span>Metric</span>
              {data.periods.map((period) => (
                <span key={period.id} className="text-right">
                  {period.label}
                </span>
              ))}
            </div>
            <PerformanceRow
              label={data.symbol}
              values={data.periods.map((period) => period.symbol_return)}
            />
            <PerformanceRow
              label={data.benchmark_label}
              values={data.periods.map((period) => period.benchmark_return)}
            />
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-500">No performance data available.</p>
      )}
    </div>
  );
}

function BenchmarkItem({
  value,
  label,
  description,
  selected,
  onSelect,
}: {
  value: string;
  label: string;
  description: string;
  selected: boolean;
  onSelect: (value: string) => void;
}) {
  return (
    <DropdownMenuItem
      onSelect={() => onSelect(value)}
      className="flex items-start gap-2"
    >
      <span className="mt-0.5 flex h-3.5 w-3.5 items-center justify-center">
        {selected && <Check className="h-3 w-3" />}
      </span>
      <span className="min-w-0">
        <span className="font-medium">{label}</span>{" "}
        <span className="text-xs text-muted-foreground">{description}</span>
      </span>
    </DropdownMenuItem>
  );
}

function LoadingTable() {
  return (
    <div className="overflow-x-auto rounded-md border border-white/10">
      <div className="min-w-[760px]">
        <div className="grid grid-cols-9 border-b border-white/10 bg-white/[0.03] px-3 py-2">
          <div className="h-3 w-12 animate-pulse rounded bg-muted" />
          {LOADING_ROWS.map((label) => (
            <span key={label} className="text-right text-xs text-slate-500">
              {label}
            </span>
          ))}
        </div>
        {["Symbol", "Benchmark"].map((label) => (
          <div
            key={label}
            className="grid grid-cols-9 border-b border-white/10 px-3 py-2 last:border-b-0"
          >
            <span className="text-sm text-slate-500">{label}</span>
            {LOADING_ROWS.map((period) => (
              <div
                key={`${label}-${period}`}
                className="ml-auto h-4 w-14 animate-pulse rounded bg-muted"
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function PerformanceRow({
  label,
  values,
}: {
  label: string;
  values: Array<number | null>;
}) {
  return (
    <div className="grid grid-cols-9 border-b border-white/10 px-3 py-2 text-sm last:border-b-0">
      <span className="text-slate-400">{label}</span>
      {values.map((value, index) => (
        <PercentValue key={index} value={value} />
      ))}
    </div>
  );
}

function PercentValue({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="text-right text-slate-500">N/A</span>;
  }

  return (
    <span
      className={cn(
        "text-right font-medium tabular-nums",
        value > 0 && "text-emerald-500",
        value < 0 && "text-red-500",
        value === 0 && "text-slate-200",
      )}
    >
      {value > 0 ? "+" : ""}
      {(value * 100).toFixed(2)}%
    </span>
  );
}
