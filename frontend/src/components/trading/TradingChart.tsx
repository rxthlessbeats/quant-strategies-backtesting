"use client";

import { useEffect, useRef } from "react";
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

export default function TradingChart({
  data,
  colorMap,
  selectedView,
  height = 520,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

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
    candleSeries.setData(toCandlestickData(data.bars));

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    volumeSeries.setData(toVolumeData(data.bars));

    const indicatorKeys = Object.keys(data.indicators);
    indicatorKeys.forEach((key) => {
      const lineSeries = chart.addSeries(LineSeries, {
        color: colorMap[key] ?? "#2962FF",
        lineWidth: 2,
        title: "",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      lineSeries.setData(
        toIndicatorLineData(data.bars, data.indicators[key] ?? []),
      );
    });

    const visibleRange = visibleRangeForView(data, selectedView);
    if (visibleRange) {
      chart.timeScale().setVisibleRange(visibleRange);
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
  }, [data, colorMap, selectedView, height]);

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
