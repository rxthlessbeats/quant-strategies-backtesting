"use client";

import { useEffect, useMemo, useRef } from "react";
import { useTheme } from "next-themes";
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
import { chartViewStart, type ChartView } from "@/lib/chart-view";
import { indicatorPane, INDICATOR_PANES } from "@/lib/indicator-utils";
import type { AnalysisChartResponse } from "@/lib/types";

interface TradingChartProps {
  data: AnalysisChartResponse;
  colorMap: Record<string, string>;
  selectedView: ChartView;
  height?: number;
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

  const from = Math.max(first, chartViewStart(last, selectedView));
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

function indicatorColor(key: string, colorMap: Record<string, string>): string {
  return colorMap[key] ?? "#2962FF";
}

function chartAppearance(dark: boolean) {
  return {
    layout: {
      background: { type: ColorType.Solid, color: dark ? "#1c1a28" : "#ffffff" },
      textColor: dark ? "#b7b3cb" : "#6b677d",
      fontFamily: window.getComputedStyle(document.body).fontFamily,
    },
    grid: {
      vertLines: { color: dark ? "#302c43" : "#f1eff6" },
      horzLines: { color: dark ? "#302c43" : "#f1eff6" },
    },
    rightPriceScale: { borderColor: dark ? "#403950" : "#e5e1ed" },
    timeScale: { borderColor: dark ? "#403950" : "#e5e1ed" },
  };
}

export default function TradingChart({
  data,
  colorMap,
  selectedView,
  height = 520,
}: TradingChartProps) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleData = useMemo(() => toCandlestickData(data.bars), [data.bars]);
  const volumeData = useMemo(() => toVolumeData(data.bars), [data.bars]);
  const indicatorKeys = useMemo(
    () => Object.keys(data.indicators),
    [data.indicators],
  );
  const paneCount = new Set(indicatorKeys.map(indicatorPane).filter(Boolean)).size;
  const chartHeight = height + paneCount * 160;
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
        indicatorKeys.filter(key => key.startsWith("macd_hist_")).map((key) => [
          key,
          toIndicatorHistogramData(data.bars, data.indicators[key] ?? []),
        ]),
      ),
    [data.bars, data.indicators, indicatorKeys],
  );
  useEffect(() => {
    const container = containerRef.current;
    if (!container || data.bars.length === 0) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: chartHeight,
      timeScale: {
        fixLeftEdge: true,
        fixRightEdge: true,
        rightOffset: 0,
      },
    });
    chart.applyOptions(chartAppearance(document.documentElement.classList.contains("dark")));
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

    const paneIndices = new Map<string, number>();
    indicatorKeys.forEach((key) => {
      const pane = indicatorPane(key);
      if (pane && !paneIndices.has(pane)) paneIndices.set(pane, paneIndices.size + 1);
      const paneIndex = pane ? paneIndices.get(pane)! : 0;
      if (key.startsWith("macd_hist_")) {
        const histogram = chart.addSeries(HistogramSeries, {
          title: "MACD histogram", lastValueVisible: true, priceLineVisible: false,
        }, paneIndex);
        histogram.setData(indicatorHistogramData[key] ?? []);
      } else {
        const precise = pane === "momentum" || pane === "cmf";
        const series = chart.addSeries(LineSeries, {
          color: indicatorColor(key, colorMap),
          lineWidth: 2,
          title: pane ? key.replaceAll("_", " ").toUpperCase() : "",
          lastValueVisible: true,
          priceLineVisible: false,
          priceFormat: pane === "obv" || pane === "ad" ? { type: "volume" }
            : { type: "price", precision: precise ? 4 : 2, minMove: precise ? .0001 : .01 },
        }, paneIndex);
        series.setData(indicatorLineData[key] ?? []);
      }
    });
    paneIndices.forEach((paneIndex, pane) => {
      INDICATOR_PANES[pane].forEach(level => {
        const guide = chart.addSeries(LineSeries, {
          color: "rgba(148, 163, 184, 0.45)", lineWidth: 1,
          lastValueVisible: false, priceLineVisible: false,
        }, paneIndex);
        guide.setData(constantLineData(data, level));
      });
    });
    chart.panes().forEach((pane, index) => pane.setStretchFactor(index === 0 ? height : 160));

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
    chartHeight,
    data,
    indicatorKeys,
    volumeData,
  ]);

  useEffect(() => {
    chartRef.current?.applyOptions(chartAppearance(dark));
  }, [dark]);

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
        className="flex items-center justify-center bg-card text-muted-foreground"
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
      role="group"
      aria-label={`${data.symbol} candlestick chart with volume and ${Object.keys(data.indicators).length} indicator series in ${paneCount} separate indicator panes. Daily prices can also be explored in the overview timeline.`}
      style={{ height: chartHeight }}
    />
  );
}
