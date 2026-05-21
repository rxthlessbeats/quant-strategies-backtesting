"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
} from "lightweight-charts";
import {
  INDICATOR_COLORS,
  toCandlestickData,
  toIndicatorLineData,
  toVolumeData,
} from "@/lib/chart-data";
import type { AnalysisChartResponse } from "@/lib/types";

interface TradingChartProps {
  data: AnalysisChartResponse;
  height?: number;
}

export default function TradingChart({ data, height = 480 }: TradingChartProps) {
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
      timeScale: { borderColor: "#334155" },
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
    indicatorKeys.forEach((key, index) => {
      const lineSeries = chart.addSeries(LineSeries, {
        color: INDICATOR_COLORS[index % INDICATOR_COLORS.length],
        lineWidth: 2,
        title: key,
      });
      lineSeries.setData(
        toIndicatorLineData(data.bars, data.indicators[key] ?? []),
      );
    });

    chart.timeScale().fitContent();

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
  }, [data, height]);

  if (data.bars.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-border bg-slate-950 text-muted-foreground"
        style={{ height }}
      >
        No bars returned for this range.
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="w-full overflow-hidden rounded-lg border border-border"
      style={{ height }}
    />
  );
}
