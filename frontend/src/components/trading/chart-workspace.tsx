"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  fetchChart,
  fetchCompanyOverview,
  fetchMarketDataArea,
  fetchPerformanceComparison,
} from "@/lib/api";
import type { ChartView } from "@/lib/chart-view";
import { latestQuoteFromBars } from "@/lib/quote-from-bars";
import {
  addSelection,
  buildColorMap,
  buildIndicatorsQuery,
  parseIndicatorSelections,
  removeSelection,
  updateSelectionParams,
  type IndicatorSelection,
} from "@/lib/indicator-utils";
import type {
  AnalysisChartResponse,
  CompanyOverview,
  MarketDataAreaResponse,
  PerformanceComparisonResponse,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import AnalystRecommendationsPanel from "./analyst-recommendations-panel";
import ChartLegend from "./chart-legend";
import ChartToolbar from "./chart-toolbar";
import CompanyStatsPanel from "./company-stats-panel";
import CompanySummary from "./company-summary";
import IndicatorSettingsPanel, {
  type SettingsMode,
} from "./indicator-settings-panel";
import MarketStatisticsPanel from "./market-statistics-panel";
import PerformanceComparisonPanel from "./performance-comparison-panel";
import TradingChart from "./TradingChart";
import ValuationMetricsPanel from "./valuation-metrics-panel";

type SettingsState =
  | { mode: "add"; id: string; defaultParams: Record<string, number> }
  | { mode: "edit"; slotId: string };

const CHART_SETTINGS_STORAGE_KEY = "rookie-trader-chart-settings";
const DEFAULT_CHART_SYMBOL = "NVDA";

interface SavedChartSettings {
  indicators: string;
}

function readSavedChartSettings(): SavedChartSettings | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(CHART_SETTINGS_STORAGE_KEY);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<SavedChartSettings>;
    if (typeof parsed.indicators !== "string") return null;
    const indicators = parsed.indicators.trim();
    if (!indicators) return null;
    return { indicators };
  } catch {
    return null;
  }
}

export default function ChartWorkspace() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const initialSymbol = searchParams.get("symbol") ?? DEFAULT_CHART_SYMBOL;
  const initialIndicators =
    searchParams.get("indicators") ?? "sma:5,sma:20";

  const [symbol, setSymbol] = useState(initialSymbol.trim().toUpperCase());
  const [selectedView, setSelectedView] = useState<ChartView>("6M");
  const [selections, setSelections] = useState<IndicatorSelection[]>(() =>
    parseIndicatorSelections(initialIndicators),
  );
  const [chartData, setChartData] = useState<AnalysisChartResponse | null>(
    null,
  );
  const [overview, setOverview] = useState<CompanyOverview | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [marketStats, setMarketStats] = useState<MarketDataAreaResponse | null>(
    null,
  );
  const [marketStatements, setMarketStatements] =
    useState<MarketDataAreaResponse | null>(null);
  const [marketEarnings, setMarketEarnings] =
    useState<MarketDataAreaResponse | null>(null);
  const [marketStatsLoading, setMarketStatsLoading] = useState(false);
  const [marketStatsError, setMarketStatsError] = useState<string | null>(null);
  const [analystData, setAnalystData] = useState<MarketDataAreaResponse | null>(
    null,
  );
  const [analystLoading, setAnalystLoading] = useState(false);
  const [analystError, setAnalystError] = useState<string | null>(null);
  const [performance, setPerformance] =
    useState<PerformanceComparisonResponse | null>(null);
  const [performanceBenchmark, setPerformanceBenchmark] = useState("SPY");
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [alertMessage, setAlertMessage] = useState<string | null>(null);
  const [alertType, setAlertType] = useState<"error" | "success">("error");
  const [alertClosing, setAlertClosing] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsState, setSettingsState] = useState<SettingsState | null>(
    null,
  );
  const [settingsAnchor, setSettingsAnchor] = useState<HTMLElement | null>(
    null,
  );
  const requestId = useRef(0);
  const ignoreSettingsCloseUntil = useRef(0);
  const alertDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismissAlert = useCallback(() => {
    if (!alertMessage) return;
    setAlertClosing(true);
    if (alertDismissTimer.current) {
      clearTimeout(alertDismissTimer.current);
    }
    alertDismissTimer.current = setTimeout(() => {
      setAlertMessage(null);
      setAlertType("error");
      setAlertClosing(false);
      alertDismissTimer.current = null;
    }, 200);
  }, [alertMessage]);

  const showAlert = useCallback(
    (message: string, type: "error" | "success" = "error") => {
      if (alertDismissTimer.current) {
        clearTimeout(alertDismissTimer.current);
        alertDismissTimer.current = null;
      }
      setAlertClosing(false);
      setAlertType(type);
      setAlertMessage(message);
      if (type === "success") {
        alertDismissTimer.current = setTimeout(() => {
          setAlertClosing(true);
          alertDismissTimer.current = setTimeout(() => {
            setAlertMessage(null);
            setAlertType("error");
            setAlertClosing(false);
            alertDismissTimer.current = null;
          }, 200);
        }, 2500);
      }
    },
    [],
  );

  useEffect(() => {
    return () => {
      if (alertDismissTimer.current) {
        clearTimeout(alertDismissTimer.current);
      }
    };
  }, []);

  useEffect(() => {
    if (searchParams.has("symbol") || searchParams.has("indicators")) return;

    const saved = readSavedChartSettings();
    if (!saved) return;

    setSelections(parseIndicatorSelections(saved.indicators));
  }, [searchParams]);

  useEffect(() => {
    const urlSymbol = searchParams.get("symbol");
    if (urlSymbol) {
      const nextSymbol = urlSymbol.trim().toUpperCase();
      setSymbol((current) => (current === nextSymbol ? current : nextSymbol));
    }
    const urlIndicators = searchParams.get("indicators");
    if (urlIndicators) {
      const nextSelections = parseIndicatorSelections(urlIndicators);
      setSelections((current) =>
        buildIndicatorsQuery(current) === buildIndicatorsQuery(nextSelections)
          ? current
          : nextSelections,
      );
    }
  }, [searchParams]);

  const colorMap = useMemo(() => buildColorMap(selections), [selections]);
  const quote = useMemo(
    () => latestQuoteFromBars(chartData?.bars ?? []),
    [chartData],
  );
  const quoteLoading =
    loading &&
    chartData?.symbol?.trim().toUpperCase() !== symbol.trim().toUpperCase();
  const showChartHeader = symbol.trim().length > 0;

  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    setSettingsState(null);
    setSettingsAnchor(null);
  }, []);

  const openAddSettings = useCallback(
    (id: string, defaultParams: Record<string, number>, anchor: HTMLElement) => {
      ignoreSettingsCloseUntil.current = Date.now() + 250;
      setSettingsState({ mode: "add", id, defaultParams });
      setSettingsAnchor(anchor);
      setSettingsOpen(true);
    },
    [],
  );

  const openEditSettings = useCallback(
    (slotId: string, anchor: HTMLElement) => {
      setSettingsState({ mode: "edit", slotId });
      setSettingsAnchor(anchor);
      setSettingsOpen(true);
    },
    [],
  );

  const settingsIndicatorId = useMemo(() => {
    if (!settingsState) return "";
    if (settingsState.mode === "add") return settingsState.id;
    return (
      selections.find((s) => s.slotId === settingsState.slotId)?.id ?? ""
    );
  }, [settingsState, selections]);

  const settingsInitialParams = useMemo(() => {
    if (!settingsState) return {};
    if (settingsState.mode === "add") return settingsState.defaultParams;
    return (
      selections.find((s) => s.slotId === settingsState.slotId)?.params ?? {}
    );
  }, [settingsState, selections]);

  const settingsMode: SettingsMode = settingsState?.mode ?? "edit";
  const settingsSlotId =
    settingsState?.mode === "edit" ? settingsState.slotId : undefined;

  const syncUrl = useCallback(
    (sym: string, indicators: string) => {
      if (!sym) return;
      const params = new URLSearchParams({
        symbol: sym,
      });
      if (indicators) {
        params.set("indicators", indicators);
      }
      if (
        window.location.pathname === "/chart" &&
        window.location.search === `?${params.toString()}`
      ) {
        return;
      }
      router.replace(`/chart?${params.toString()}`, { scroll: false });
    },
    [router],
  );

  const saveChartSettings = useCallback(() => {
    window.localStorage.setItem(
      CHART_SETTINGS_STORAGE_KEY,
      JSON.stringify({
        indicators: buildIndicatorsQuery(selections),
      }),
    );
    showAlert("Preset saved", "success");
  }, [selections, showAlert]);

  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    const indicators = buildIndicatorsQuery(selections);

    if (!sym) {
      requestId.current += 1;
      setChartData(null);
      setLoading(false);
      setError(null);
      return;
    }

    const timer = setTimeout(() => {
      const id = ++requestId.current;
      setLoading(true);
      setError(null);

      fetchChart({
        symbol: sym,
        interval: "1d",
        indicators: indicators || undefined,
      })
        .then((data) => {
          if (id !== requestId.current) return;
          setChartData(data);
          syncUrl(sym, indicators);
        })
        .catch((e) => {
          if (id !== requestId.current) return;
          setError(e instanceof Error ? e.message : "Request failed");
        })
        .finally(() => {
          if (id === requestId.current) setLoading(false);
        });
    }, 350);

    return () => clearTimeout(timer);
  }, [symbol, selections, syncUrl]);

  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    if (!sym) {
      setOverview(null);
      return;
    }

    let cancelled = false;
    setOverviewLoading(true);
    fetchCompanyOverview(sym)
      .then((data) => {
        if (!cancelled) setOverview(data);
      })
      .catch(() => {
        if (!cancelled) setOverview(null);
      })
      .finally(() => {
        if (!cancelled) setOverviewLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    if (!sym) {
      setAnalystData(null);
      setAnalystLoading(false);
      setAnalystError(null);
      return;
    }

    let cancelled = false;
    setAnalystLoading(true);
    setAnalystError(null);
    fetchMarketDataArea(sym, "analysts")
      .then((data) => {
        if (!cancelled) setAnalystData(data);
      })
      .catch((e) => {
        if (cancelled) return;
        setAnalystData(null);
        setAnalystError(
          e instanceof Error ? e.message : "Failed to load analyst data",
        );
      })
      .finally(() => {
        if (!cancelled) setAnalystLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    if (!sym) {
      setMarketStats(null);
      setMarketStatements(null);
      setMarketEarnings(null);
      setMarketStatsLoading(false);
      setMarketStatsError(null);
      return;
    }

    let cancelled = false;
    setMarketStatsLoading(true);
    setMarketStatements(null);
    setMarketEarnings(null);
    setMarketStatsError(null);
    fetchMarketDataArea(sym, "statistics")
      .then((data) => {
        if (cancelled) return;
        setMarketStats(data);
      })
      .catch((e) => {
        if (cancelled) return;
        setMarketStats(null);
        setMarketStatsError(
          e instanceof Error ? e.message : "Failed to load market statistics",
        );
      })
      .finally(() => {
        if (!cancelled) setMarketStatsLoading(false);
      });

    fetchMarketDataArea(sym, "statements")
      .then((data) => {
        if (!cancelled) setMarketStatements(data);
      })
      .catch(() => {
        if (!cancelled) setMarketStatements(null);
      });

    fetchMarketDataArea(sym, "earnings")
      .then((data) => {
        if (!cancelled) setMarketEarnings(data);
      })
      .catch(() => {
        if (!cancelled) setMarketEarnings(null);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    if (!sym) {
      setPerformance(null);
      setPerformanceLoading(false);
      setPerformanceError(null);
      return;
    }

    let cancelled = false;
    setPerformanceLoading(true);
    setPerformanceError(null);
    fetchPerformanceComparison(sym, performanceBenchmark)
      .then((data) => {
        if (!cancelled) setPerformance(data);
      })
      .catch((e) => {
        if (cancelled) return;
        setPerformance(null);
        setPerformanceError(
          e instanceof Error ? e.message : "Failed to load performance",
        );
      })
      .finally(() => {
        if (!cancelled) setPerformanceLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol, performanceBenchmark]);

  const handleSettingsApply = useCallback(
    (params: Record<string, number>) => {
      if (!settingsState) return;

      if (settingsState.mode === "add") {
        setSelections((prev) =>
          addSelection(prev, settingsState.id, params),
        );
      } else {
        setSelections((prev) =>
          updateSelectionParams(prev, settingsState.slotId, params),
        );
      }
      setAlertMessage(null);
      setAlertClosing(false);
      closeSettings();
    },
    [settingsState, closeSettings],
  );

  const handleRemove = useCallback(
    (slotId: string) => {
      setSelections((prev) => removeSelection(prev, slotId));
      if (
        settingsState?.mode === "edit" &&
        settingsState.slotId === slotId
      ) {
        closeSettings();
      }
    },
    [settingsState, closeSettings],
  );

  return (
    <div className="flex flex-col gap-0">
      {alertMessage && (
        <div className="pointer-events-none fixed inset-x-0 top-4 z-50 flex justify-center px-4">
          <div
            role="alert"
            className={cn(
              "pointer-events-auto flex w-full max-w-md items-start gap-3",
              "rounded-md border px-4 py-3 text-sm shadow-lg backdrop-blur-sm",
              "transition-all duration-200 ease-out",
              alertType === "success"
                ? "border-emerald-500/50 bg-emerald-950/95 text-emerald-100"
                : "border-red-500/50 bg-red-950/95 text-red-100",
              alertClosing
                ? "-translate-y-4 opacity-0"
                : "animate-in fade-in-0 slide-in-from-top-4 duration-300",
            )}
          >
            {alertType === "success" ? (
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />
            ) : (
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-300" />
            )}
            <span className="flex-1">{alertMessage}</span>
            <button
              type="button"
              onClick={dismissAlert}
              aria-label="Dismiss alert"
              className={cn(
                "rounded p-1",
                alertType === "success"
                  ? "text-emerald-200 hover:bg-emerald-500/20"
                  : "text-red-200 hover:bg-red-500/20",
              )}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
      {showChartHeader && (
        <>
          <div
            className={cn(
              "fixed inset-x-0 top-16 z-30 border-b border-border/60 shadow-sm",
              "bg-background/80 px-6 py-3 backdrop-blur-md",
              "supports-[backdrop-filter]:bg-background/65",
              "tablet:left-44 tablet:px-10",
              "desktop:px-14",
            )}
          >
            <CompanySummary
              overview={overview}
              overviewLoading={overviewLoading}
              quote={quote}
              quoteLoading={quoteLoading}
            />
          </div>
          <div className="h-[85px]" aria-hidden="true" />
        </>
      )}
      <div className="mt-3 flex flex-col gap-3">
        <div className="overflow-hidden rounded-lg border border-border bg-black/40 backdrop-blur-md">
          <ChartToolbar
            selectedView={selectedView}
            onViewChange={setSelectedView}
            loading={loading}
            meta={chartData?.meta}
            onAddIndicatorPick={openAddSettings}
            onSaveSettings={saveChartSettings}
          />

          <div className="relative overflow-hidden border-t border-border bg-[#020817]">
            {error && (
              <div className="absolute left-3 top-2 z-10 rounded-md border border-destructive/50 bg-destructive/90 px-2 py-1 text-xs text-white">
                {error}
              </div>
            )}
            <ChartLegend
              selections={selections}
              onOpenSettings={openEditSettings}
              onRemove={handleRemove}
            />
            {chartData ? (
              <TradingChart
                data={chartData}
                colorMap={colorMap}
                selectedView={selectedView}
              />
            ) : (
              <div
                className={cn(
                  "flex items-center justify-center text-sm text-slate-500",
                  "h-[520px]",
                )}
              >
                {loading ? "Loading chart…" : "Select settings to load chart"}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-3 lg:flex-row">
          <MarketStatisticsPanel
            data={marketStats}
            bars={chartData?.bars ?? []}
            loading={marketStatsLoading}
            error={marketStatsError}
          />

          <ValuationMetricsPanel
            data={marketStats}
            statements={marketStatements}
            bars={chartData?.bars ?? []}
            loading={marketStatsLoading}
            error={marketStatsError}
          />
        </div>

        <PerformanceComparisonPanel
          data={performance}
          loading={performanceLoading}
          error={performanceError}
          benchmark={performanceBenchmark}
          onBenchmarkChange={setPerformanceBenchmark}
        />

        <AnalystRecommendationsPanel
          data={analystData}
          marketStats={marketStats}
          bars={chartData?.bars ?? []}
          loading={analystLoading}
          error={analystError}
        />

        <CompanyStatsPanel
          overview={overview}
          data={marketStats}
          statements={marketStatements}
          earnings={marketEarnings}
          loading={overviewLoading || marketStatsLoading}
        />
      </div>

      <IndicatorSettingsPanel
        open={settingsOpen && settingsState !== null}
        onOpenChange={(open) => {
          if (open) {
            setSettingsOpen(true);
            return;
          }
          if (Date.now() < ignoreSettingsCloseUntil.current) {
            return;
          }
          closeSettings();
        }}
        mode={settingsMode}
        indicatorId={settingsIndicatorId}
        initialParams={settingsInitialParams}
        slotId={settingsSlotId}
        anchorEl={settingsAnchor}
        selections={selections}
        onApply={handleSettingsApply}
        onBack={closeSettings}
        onPeriodError={showAlert}
      />
    </div>
  );
}
