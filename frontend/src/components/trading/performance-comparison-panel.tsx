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
  const [selectedPeriod, setSelectedPeriod] = useState("1y");
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

  const activePeriod = data?.periods.find(period => period.id === selectedPeriod) ?? data?.periods[0];
  const activeIndex = data?.periods.findIndex(period => period.id === activePeriod?.id) ?? -1;
  const scale = Math.max(Math.abs(activePeriod?.symbol_return ?? 0), Math.abs(activePeriod?.benchmark_return ?? 0), 0.01);

  return (
    <div className="research-section performance-section" aria-busy={loading}>
      <div className="research-heading">
        <div><h2>Performance Comparison</h2><p>See how the company moves against your benchmark.</p></div>
        {data?.as_of && !loading && <span className="research-date">As of {data.as_of}</span>}
      </div>
      <div className="performance-controls">
        <div className="benchmark-control">
          <span>Compare against</span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button type="button" variant="ghost" className="benchmark-picker" aria-label={`Choose benchmark, ${benchmarkUpper}`}>
                {selectedBenchmark?.symbol ?? benchmarkUpper}<ChevronDown size={14} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="max-h-96 w-72 overflow-y-auto">
              {benchmarkGroups.map((group, groupIndex) => (
                <div key={group.category}>
                  {groupIndex > 0 && <DropdownMenuSeparator />}
                  <DropdownMenuLabel className="text-xs text-muted-foreground">{group.category}</DropdownMenuLabel>
                  {group.options.map(option => <BenchmarkItem key={option.symbol} value={option.symbol} label={option.symbol} description={option.description} selected={option.symbol === benchmarkUpper} onSelect={chooseBenchmark} />)}
                </div>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <form onSubmit={handleCustomSubmit} className="custom-benchmark">
          <span>or</span>
          <div>
            <input type="text" value={searchInput} onChange={event => setSearchInput(event.target.value.toUpperCase())} placeholder="Enter a symbol" aria-label="Custom benchmark symbol" />
            <button type="submit" aria-label="Set custom benchmark" disabled={!searchInput.trim()}><Search size={15} /></button>
          </div>
        </form>
        <span className="comparison-status" data-loading={loading} role="status">{loading ? "Updating comparison…" : !error && data && activePeriod ? `Showing ${activePeriod.label} returns against ${data.benchmark_label}.` : ""}</span>
      </div>
      {error ? <p className="research-error">{error}</p> : loading ? (
        <div className="performance-pending"><p className="research-loading">Loading performance returns…</p><LoadingTable /></div>
      ) : data && activePeriod ? (
        <>
          <div className="performance-focus">
            <div className="comparison-readout" key={`${data.symbol}-${data.benchmark_symbol}-${activePeriod.id}`}>
              <p>{activePeriod.label} return</p>
              <dl className="comparison-values" aria-live="polite" aria-atomic="true">
                <div><dt><span className="series-key company-key" />{data.symbol}</dt><dd><PercentValue value={activePeriod.symbol_return} /></dd></div>
                <div><dt><span className="series-key benchmark-key" />{data.benchmark_label}</dt><dd><PercentValue value={activePeriod.benchmark_return} /></dd></div>
              </dl>
            </div>
            <div className="comparison-plot">
              <div className="period-selector" role="group" aria-label="Performance period">
                {data.periods.map(period => <button key={period.id} type="button" aria-label={`Compare ${period.label} returns`} aria-pressed={activePeriod.id === period.id} onClick={() => setSelectedPeriod(period.id)}>{period.label}</button>)}
              </div>
              <div className="comparison-bars" aria-hidden="true">
                {[[data.symbol, activePeriod.symbol_return], [data.benchmark_label, activePeriod.benchmark_return]].map(([label, value], index) => {
                  const number = typeof value === "number" ? value : null;
                  const width = number === null ? 0 : Math.abs(number) / scale * 47;
                  return <div className="comparison-bar-row" key={index}><span>{label}</span><div className="return-track"><i className="return-zero" /><span className={index === 0 ? "company-return" : "benchmark-return"} style={{ left: `${number !== null && number < 0 ? 50 - width : 50}%`, width: `${width}%` }} /></div></div>;
                })}
                <span className="comparison-zero-label">0%</span>
              </div>
            </div>
          </div>
          <div className="performance-scroll" tabIndex={0} role="region" aria-label="Performance returns by period">
            <table className="performance-table">
              <thead><tr><th scope="col" className="text-left">Returns</th>{data.periods.map((period, index) => <th key={period.id} scope="col" data-selected={index === activeIndex} className="text-right">{period.label}</th>)}</tr></thead>
              <tbody>
                <PerformanceRow label={data.symbol} values={data.periods.map(period => period.symbol_return)} activeIndex={activeIndex} />
                <PerformanceRow label={data.benchmark_label} values={data.periods.map(period => period.benchmark_return)} activeIndex={activeIndex} />
              </tbody>
            </table>
          </div>
          <p className="research-footnote">Returns are based on daily closing prices.</p>
        </>
      ) : <p className="research-empty">No performance data available.</p>}
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
    <div className="overflow-x-auto rounded-md border border-border" tabIndex={0} role="region" aria-label="Loading performance returns">
      <div className="min-w-[760px]">
        <div className="grid grid-cols-9 border-b border-border bg-muted/50 px-3 py-2">
          <div className="h-3 w-12 animate-pulse rounded bg-muted" />
          {LOADING_ROWS.map((label) => (
            <span key={label} className="text-right text-xs text-muted-foreground">
              {label}
            </span>
          ))}
        </div>
        {["Symbol", "Benchmark"].map((label) => (
          <div
            key={label}
            className="grid grid-cols-9 border-b border-border px-3 py-2 last:border-b-0"
          >
            <span className="text-sm text-muted-foreground">{label}</span>
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
  activeIndex,
}: {
  label: string;
  values: Array<number | null>;
  activeIndex: number;
}) {
  return (
    <tr className="text-sm">
      <th scope="row" className="text-left font-normal text-muted-foreground">{label}</th>
      {values.map((value, index) => (
        <td key={index} className="text-right" data-selected={index === activeIndex}><PercentValue value={value} /></td>
      ))}
    </tr>
  );
}

function PercentValue({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="text-right text-muted-foreground">N/A</span>;
  }

  return (
    <span
      className={cn(
        "text-right font-medium tabular-nums",
        value > 0 && "text-[var(--positive)]",
        value < 0 && "text-[var(--negative)]",
        value === 0 && "text-foreground",
      )}
    >
      {value > 0 ? "+" : ""}
      {(value * 100).toFixed(2)}%
    </span>
  );
}
