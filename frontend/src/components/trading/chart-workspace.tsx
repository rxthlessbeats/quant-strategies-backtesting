"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { fetchChart } from "@/lib/api";
import type { ChartView } from "@/lib/chart-view";
import {
  addSelection,
  buildColorMap,
  buildIndicatorsQuery,
  parseIndicatorSelections,
  removeSelection,
  updateSelectionPeriod,
  type IndicatorSelection,
} from "@/lib/indicator-utils";
import type { AnalysisChartResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import ChartLegend from "./chart-legend";
import ChartToolbar from "./chart-toolbar";
import IndicatorSettingsPanel, {
  type SettingsMode,
} from "./indicator-settings-panel";
import TradingChart from "./TradingChart";

type SettingsState =
  | { mode: "add"; id: string; defaultPeriod: number }
  | { mode: "edit"; slotId: string };

export default function ChartWorkspace() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const initialSymbol = searchParams.get("symbol") ?? "AAPL";
  const initialIndicators =
    searchParams.get("indicators") ?? "sma:5,sma:20";

  const [symbolInput, setSymbolInput] = useState(initialSymbol.toUpperCase());
  const [symbol, setSymbol] = useState(initialSymbol.trim().toUpperCase());
  const [selectedView, setSelectedView] = useState<ChartView>("1Y");
  const [selections, setSelections] = useState<IndicatorSelection[]>(() =>
    parseIndicatorSelections(initialIndicators),
  );
  const [chartData, setChartData] = useState<AnalysisChartResponse | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [alertMessage, setAlertMessage] = useState<string | null>(null);
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
      setAlertClosing(false);
      alertDismissTimer.current = null;
    }, 200);
  }, [alertMessage]);

  const showAlert = useCallback((message: string) => {
    if (alertDismissTimer.current) {
      clearTimeout(alertDismissTimer.current);
      alertDismissTimer.current = null;
    }
    setAlertClosing(false);
    setAlertMessage(message);
  }, []);

  useEffect(() => {
    return () => {
      if (alertDismissTimer.current) {
        clearTimeout(alertDismissTimer.current);
      }
    };
  }, []);

  const colorMap = useMemo(() => buildColorMap(selections), [selections]);

  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    setSettingsState(null);
    setSettingsAnchor(null);
  }, []);

  const openAddSettings = useCallback(
    (id: string, defaultPeriod: number, anchor: HTMLElement) => {
      ignoreSettingsCloseUntil.current = Date.now() + 250;
      setSettingsState({ mode: "add", id, defaultPeriod });
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

  const settingsInitialPeriod = useMemo(() => {
    if (!settingsState) return 20;
    if (settingsState.mode === "add") return settingsState.defaultPeriod;
    return (
      selections.find((s) => s.slotId === settingsState.slotId)?.period ?? 20
    );
  }, [settingsState, selections]);

  const settingsMode: SettingsMode = settingsState?.mode ?? "edit";
  const settingsSlotId =
    settingsState?.mode === "edit" ? settingsState.slotId : undefined;

  const syncUrl = useCallback(
    (sym: string, indicators: string) => {
      const params = new URLSearchParams({
        symbol: sym,
      });
      if (indicators) {
        params.set("indicators", indicators);
      }
      router.replace(`/chart?${params.toString()}`, { scroll: false });
    },
    [router],
  );

  const submitSymbol = useCallback(() => {
    const nextSymbol = symbolInput.trim().toUpperCase();
    if (!nextSymbol) return;
    setSymbolInput(nextSymbol);
    setSymbol(nextSymbol);
  }, [symbolInput]);

  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    const indicators = buildIndicatorsQuery(selections);

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

  const handleSettingsApply = useCallback(
    (period: number) => {
      if (!settingsState) return;

      if (settingsState.mode === "add") {
        setSelections((prev) =>
          addSelection(prev, settingsState.id, period),
        );
      } else {
        setSelections((prev) =>
          updateSelectionPeriod(prev, settingsState.slotId, period),
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
    <div className="flex flex-col gap-3">
      {alertMessage && (
        <div className="pointer-events-none fixed inset-x-0 top-4 z-50 flex justify-center px-4">
          <div
            role="alert"
            className={cn(
              "pointer-events-auto flex w-full max-w-md items-start gap-3",
              "rounded-md border border-red-500/50 bg-red-950/95 px-4 py-3",
              "text-sm text-red-100 shadow-lg backdrop-blur-sm",
              "transition-all duration-200 ease-out",
              alertClosing
                ? "-translate-y-4 opacity-0"
                : "animate-in fade-in-0 slide-in-from-top-4 duration-300",
            )}
          >
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-300" />
            <span className="flex-1">{alertMessage}</span>
            <button
              type="button"
              onClick={dismissAlert}
              aria-label="Dismiss alert"
              className="rounded p-1 text-red-200 hover:bg-red-500/20"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
      <div className="rounded-lg border border-border bg-black/40 backdrop-blur-md">
        <ChartToolbar
          symbol={symbolInput}
          onSymbolChange={setSymbolInput}
          onSymbolSubmit={submitSymbol}
          selectedView={selectedView}
          onViewChange={setSelectedView}
          loading={loading}
          meta={chartData?.meta}
          onAddIndicatorPick={openAddSettings}
        />
      </div>

      <div className="relative overflow-hidden rounded-lg border border-border bg-[#020817]">
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
        initialPeriod={settingsInitialPeriod}
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
