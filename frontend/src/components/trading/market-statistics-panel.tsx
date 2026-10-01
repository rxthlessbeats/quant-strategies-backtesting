"use client";

import { formatDailyMarketAsOf } from "@/lib/market-timestamps";
import type { BarPoint, MarketDataAreaResponse } from "@/lib/types";

interface MarketStatisticsPanelProps {
  data: MarketDataAreaResponse | null;
  bars: BarPoint[];
  loading: boolean;
  error: string | null;
}

interface YahooValue {
  raw?: unknown;
  fmt?: unknown;
}

const TRADING_DAYS_PER_YEAR = 252;
const VOLUME_WINDOW_DAYS = 20;

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

function rangePosition(
  low: number | null,
  high: number | null,
  value: number | null,
): number {
  if (low == null || high == null || value == null || high <= low) return 0;
  return Math.min(1, Math.max(0, (value - low) / (high - low)));
}

function VolumeRangeChart({
  low,
  average,
  today,
  high,
}: {
  low: number | null;
  average: number | null;
  today: number | null;
  high: number | null;
}) {
  const avgPosition = rangePosition(low, high, average);
  const todayPosition = rangePosition(low, high, today);
  const hasRange = low != null && high != null && high > low;

  return (
    <div className="volume-profile">
      <h4>Trading volume</h4>
      <div className="volume-range">
        {hasRange && (
          <>
            <div
              className="volume-today"
              style={{ width: `${todayPosition * 100}%` }}
            />
            <div
              className="volume-average"
              style={{ left: `${avgPosition * 100}%` }}
              title={`20D avg ${formatNumber(average)}`}
            />
          </>
        )}
      </div>
      <div className="range-endpoints">
        <span>20D low {formatNumber(low)}</span>
        <span>20D high {formatNumber(high)}</span>
      </div>
      <dl className="volume-readout">
        <div>
          <dt>Today</dt>
          <dd>{formatNumber(today)}</dd>
        </div>
        <div>
          <dt><span className="average-key" />20D average</dt>
          <dd>{formatNumber(average)}</dd>
        </div>
      </dl>
    </div>
  );
}

function OhlcSegmentChart({
  label,
  low,
  open,
  close,
  high,
  lowLabel = "Low",
  highLabel = "High",
}: {
  label: string;
  low: number | null;
  open: number | null;
  close: number | null;
  high: number | null;
  lowLabel?: string;
  highLabel?: string;
}) {
  const openPosition = rangePosition(low, high, open);
  const closePosition = rangePosition(low, high, close);
  const hasRange = low != null && high != null && high > low;
  const hasSegments = hasRange && open != null && close != null;
  const isDown = open != null && close != null && close < open;
  const lowerPosition = Math.min(openPosition, closePosition);
  const upperPosition = Math.max(openPosition, closePosition);
  const startSegmentWidth = lowerPosition * 100;
  const changeSegmentWidth = (upperPosition - lowerPosition) * 100;
  const endSegmentWidth = (1 - upperPosition) * 100;
  const segmentStyle = (width: number) => ({
    flexBasis: 0,
    flexGrow: Math.max(width, 0),
  });

  return (
    <div className="market-range">
      <p>{label}</p>
      <div className="market-range-track">
        {hasSegments && (
          <>
            <div
              className="range-remainder"
              style={segmentStyle(startSegmentWidth)}
            />
            <div
              className={`range-session ${isDown ? "is-down" : "is-up"}`}
              style={segmentStyle(changeSegmentWidth)}
            />
            <div
              className="range-remainder"
              style={segmentStyle(endSegmentWidth)}
            />
          </>
        )}
      </div>
      <div className="range-endpoints">
        <span>
          {lowLabel} {formatNumber(low)}
        </span>
        <span>
          {highLabel} {formatNumber(high)}
        </span>
      </div>
    </div>
  );
}

function formattedValue(value: unknown, fallback = "—"): string {
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

function percentValue(value: unknown): string {
  const raw = rawNumber(value);
  if (raw == null) return formattedValue(value);
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 2,
  }).format(raw);
}

function modulePayload(data: MarketDataAreaResponse | null, module: string) {
  return data?.modules.find((item) => item.module === module)?.payload ?? null;
}

function formatNumber(value: number | null): string {
  return value == null ? "—" : formattedValue(value);
}

function getVolumeStatsFromBars(bars: BarPoint[]) {
  const windowBars = bars.slice(-VOLUME_WINDOW_DAYS);
  if (!windowBars.length) {
    return null;
  }

  const volumes = windowBars.map((bar) => bar.volume);
  const low = Math.min(...volumes);
  const high = Math.max(...volumes);
  const average = volumes.reduce((sum, volume) => sum + volume, 0) / volumes.length;
  const today = windowBars[windowBars.length - 1]?.volume ?? null;

  return { low, high, average, today };
}

function getRangeFromBars(bars: BarPoint[]) {
  const windowBars = bars.slice(-TRADING_DAYS_PER_YEAR);
  if (!windowBars.length) {
    return null;
  }

  let lowBar = windowBars[0];
  let highBar = windowBars[0];
  for (const bar of windowBars) {
    if (bar.low < lowBar.low) lowBar = bar;
    if (bar.high > highBar.high) highBar = bar;
  }

  const latest = windowBars[windowBars.length - 1];
  const position =
    highBar.high > lowBar.low
      ? Math.min(1, Math.max(0, (latest.close - lowBar.low) / (highBar.high - lowBar.low)))
      : 0;
  const drawdownFromHigh =
    highBar.high > 0 ? Math.max(0, (highBar.high - latest.close) / highBar.high) : null;

  return {
    low: lowBar.low,
    lowDate: lowBar.timestamp,
    high: highBar.high,
    highDate: highBar.timestamp,
    latestOpen: latest.open,
    latestHigh: latest.high,
    latestLow: latest.low,
    latestClose: latest.close,
    latestDate: latest.timestamp,
    position,
    drawdownFromHigh,
  };
}

export default function MarketStatisticsPanel({
  data,
  bars,
  loading,
  error,
}: MarketStatisticsPanelProps) {
  const detail = modulePayload(data, "summaryDetail");
  const barRange = getRangeFromBars(bars);
  const yahooLow = rawNumber(detail?.fiftyTwoWeekLow);
  const yahooHigh = rawNumber(detail?.fiftyTwoWeekHigh);
  const yahooClose =
    rawNumber(detail?.regularMarketPreviousClose) ?? rawNumber(detail?.previousClose);
  const rangeLow = barRange?.low ?? yahooLow;
  const rangeHigh = Math.max(
    barRange?.high ?? Number.NEGATIVE_INFINITY,
    yahooHigh ?? Number.NEGATIVE_INFINITY,
  );
  const close = barRange?.latestClose ?? yahooClose;
  const normalizedRangeHigh = Number.isFinite(rangeHigh) ? rangeHigh : null;
  const drawdownFromHigh =
    close != null && normalizedRangeHigh != null && normalizedRangeHigh > 0
      ? Math.max(0, (normalizedRangeHigh - close) / normalizedRangeHigh)
      : barRange?.drawdownFromHigh ?? null;
  const volumeStats = getVolumeStatsFromBars(bars);
  const yahooTodayVolume = rawNumber(detail?.volume);
  const yahooAverageVolume =
    rawNumber(detail?.averageVolume) ??
    rawNumber(detail?.averageDailyVolume10Day);
  const volumeLow = volumeStats?.low ?? null;
  const volumeHigh = volumeStats?.high ?? null;
  const volumeAverage = volumeStats?.average ?? yahooAverageVolume;
  const volumeToday = volumeStats?.today ?? yahooTodayVolume;
  const updatedAt = formatDailyMarketAsOf(barRange?.latestDate);

  return (
    <section className="market-statistics">
      <div className="panel-subheading">
        <h3>Market statistics</h3>
        {updatedAt && <span>Daily close · {updatedAt}</span>}
      </div>
      {error ? <p className="research-error">{error}</p> : loading ? (
        <p className="research-loading" role="status">Loading market statistics…</p>
      ) : (
        <>
          <div className="market-price-ranges">
            <h4>Price range</h4>
            <OhlcSegmentChart label="Today" low={barRange?.latestLow ?? null} open={barRange?.latestOpen ?? null} close={close} high={barRange?.latestHigh ?? null} />
            <OhlcSegmentChart label="52 weeks" low={rangeLow} open={barRange?.latestOpen ?? null} close={close} high={normalizedRangeHigh} />
          </div>
          <dl className="market-session">
            {[
              ["Today high", formatNumber(barRange?.latestHigh ?? null)],
              ["Today low", formatNumber(barRange?.latestLow ?? null)],
              ["Today open", formatNumber(barRange?.latestOpen ?? null)],
              ["Today close", formatNumber(close)],
              ["Drop from high", drawdownFromHigh == null ? "—" : percentValue(-drawdownFromHigh)],
            ].map(([label, value]) => (
              <div key={label}><dt>{label}</dt><dd className={label === "Drop from high" ? "negative" : undefined}>{value}</dd></div>
            ))}
          </dl>
          <VolumeRangeChart low={volumeLow} average={volumeAverage} today={volumeToday} high={volumeHigh} />
        </>
      )}
    </section>
  );
}
