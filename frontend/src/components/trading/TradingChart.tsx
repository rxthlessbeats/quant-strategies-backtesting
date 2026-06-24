"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
  type UTCTimestamp,
} from "lightweight-charts";
import {
  toCandlestickData,
  toIndicatorHistogramData,
  toIndicatorLineData,
  toVolumeData,
} from "@/lib/chart-data";
import type { ChartView } from "@/lib/chart-view";
import type { AnalysisChartResponse } from "@/lib/types";

interface TradingChartProps {
  data: AnalysisChartResponse;
  colorMap: Record<string, string>;
  selectedView: ChartView;
  height?: number;
}

const VIEW_MONTHS: Partial<Record<ChartView, number>> = {
  "1M": 1,
  "3M": 3,
  "6M": 6,
  "1Y": 12,
  "5Y": 60,
  "10Y": 120,
};

function subtractMonths(timestamp: number, months: number): UTCTimestamp {
  const date = new Date(timestamp * 1000);
  date.setUTCMonth(date.getUTCMonth() - months);
  return Math.floor(date.getTime() / 1000) as UTCTimestamp;
}

function visibleRangeForView(
  data: AnalysisChartResponse,
  selectedView: ChartView,
) {
  const first = data.bars[0]?.timestamp;
  const last = data.bars[data.bars.length - 1]?.timestamp;
  if (!first || !last) return null;

  if (selectedView === "ALL") {
    return {
      from: first as UTCTimestamp,
      to: last as UTCTimestamp,
    };
  }

  const months = VIEW_MONTHS[selectedView] ?? 12;
  const from = Math.max(first, subtractMonths(last, months));
  return {
    from: from as UTCTimestamp,
    to: last as UTCTimestamp,
  };
}

function constantLineData(data: AnalysisChartResponse, value: number) {
  return data.bars.map((bar) => ({
    time: bar.timestamp as UTCTimestamp,
    value,
  }));
}

function isMacdKey(key: string): boolean {
  return key.startsWith("macd_");
}

function isRsiKey(key: string): boolean {
  return key.startsWith("rsi_");
}

function indicatorColor(key: string, colorMap: Record<string, string>): string {
  return colorMap[key] ?? "#2962FF";
}

export default function TradingChart({
  data,
  colorMap,
  selectedView,
  height = 520,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleData = useMemo(() => toCandlestickData(data.bars), [data.bars]);
  const volumeData = useMemo(() => toVolumeData(data.bars), [data.bars]);
  const indicatorKeys = useMemo(
    () => Object.keys(data.indicators),
    [data.indicators],
  );
  const rsiKeys = useMemo(() => indicatorKeys.filter(isRsiKey), [indicatorKeys]);
  const macdKeys = useMemo(
    () => indicatorKeys.filter(isMacdKey),
    [indicatorKeys],
  );
  const priceKeys = useMemo(
    () => indicatorKeys.filter((key) => !isRsiKey(key) && !isMacdKey(key)),
    [indicatorKeys],
  );
  const indicatorLineData = useMemo(
    () =>
      Object.fromEntries(
        indicatorKeys.map((key) => [
          key,
          toIndicatorLineData(data.bars, data.indicators[key] ?? []),
        ]),
      ),
    [data.bars, data.indicators, indicatorKeys],
  );
  const indicatorHistogramData = useMemo(
    () =>
      Object.fromEntries(
        indicatorKeys.map((key) => [
          key,
          toIndicatorHistogramData(data.bars, data.indicators[key] ?? []),
        ]),
      ),
    [data.bars, data.indicators, indicatorKeys],
  );
  const rsiGuideData = useMemo(
    () => ({
      70: constantLineData(data, 70),
      30: constantLineData(data, 30),
    }),
    [data],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container || data.bars.length === 0) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#020817" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
      rightPriceScale: { borderColor: "#334155" },
      timeScale: {
        borderColor: "#334155",
        fixLeftEdge: true,
        fixRightEdge: true,
        rightOffset: 0,
      },
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });
    candleSeries.setData(candleData);

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    volumeSeries.setData(volumeData);

    let nextPaneIndex = 1;
    const rsiPaneIndex = rsiKeys.length > 0 ? nextPaneIndex++ : null;
    const macdPaneIndex = macdKeys.length > 0 ? nextPaneIndex++ : null;

    priceKeys.forEach((key) => {
      const lineSeries = chart.addSeries(LineSeries, {
        color: indicatorColor(key, colorMap),
        lineWidth: 2,
        title: "",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      lineSeries.setData(indicatorLineData[key] ?? []);
    });

    if (rsiPaneIndex !== null) {
      rsiKeys.forEach((key) => {
        const lineSeries = chart.addSeries(
          LineSeries,
          {
            color: indicatorColor(key, colorMap),
            lineWidth: 2,
            title: "",
            lastValueVisible: true,
            priceLineVisible: false,
          },
          rsiPaneIndex,
        );
        lineSeries.setData(indicatorLineData[key] ?? []);
      });

      [70, 30].forEach((level) => {
        const guideSeries = chart.addSeries(
          LineSeries,
          {
            color: "rgba(148, 163, 184, 0.45)",
            lineWidth: 1,
            title: "",
            lastValueVisible: false,
            priceLineVisible: false,
          },
          rsiPaneIndex,
        );
        guideSeries.setData(rsiGuideData[level as 70 | 30]);
      });
    }

    if (macdPaneIndex !== null) {
      macdKeys.forEach((key) => {
        if (key.includes("_hist_")) {
          const histSeries = chart.addSeries(
            HistogramSeries,
            {
              title: "",
              lastValueVisible: true,
              priceLineVisible: false,
            },
            macdPaneIndex,
          );
          histSeries.setData(indicatorHistogramData[key] ?? []);
          return;
        }

        const lineSeries = chart.addSeries(
          LineSeries,
          {
            color: indicatorColor(key, colorMap),
            lineWidth: 2,
            title: "",
            lastValueVisible: true,
            priceLineVisible: false,
          },
          macdPaneIndex,
        );
        lineSeries.setData(indicatorLineData[key] ?? []);
      });
    }

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0) {
          chart.applyOptions({ width: entry.contentRect.width });
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [
    data.bars.length,
    candleData,
    colorMap,
    height,
    indicatorHistogramData,
    indicatorLineData,
    macdKeys,
    priceKeys,
    rsiGuideData,
    rsiKeys,
    volumeData,
  ]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || data.bars.length === 0) return;

    const visibleRange = visibleRangeForView(data, selectedView);
    if (visibleRange) {
      chart.timeScale().setVisibleRange(visibleRange);
    }
  }, [data, selectedView]);

  if (data.bars.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-slate-950 text-muted-foreground"
        style={{ height }}
      >
        No bars returned for this range.
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-full w-full"
      style={{ height }}
    />
  );
}
