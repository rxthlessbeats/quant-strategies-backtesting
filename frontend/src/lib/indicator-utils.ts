export interface IndicatorSelection {
  slotId: string;
  id: string;
  params: Record<string, number>;
}

export function buildIndicatorsQuery(
  selections: IndicatorSelection[],
): string {
  return selections.map(selectionQueryPart).join(",");
}

export function selectionApiKey(sel: IndicatorSelection): string {
  const suffix = paramsSuffix(sel.id, sel.params);
  return suffix ? `${sel.id}_${suffix}` : sel.id;
}

export function hasDuplicate(
  selections: IndicatorSelection[],
  id: string,
  params: Record<string, number>,
): boolean {
  const idNorm = id.toLowerCase();
  return selections.some(
    (s) => s.id === idNorm && paramsEqual(s.params, params),
  );
}

export function createSelection(
  id: string,
  params: Record<string, number>,
): IndicatorSelection {
  const idNorm = id.toLowerCase();
  const signature = paramsSuffix(idNorm, params);
  return {
    slotId: `${idNorm}-${signature}-${crypto.randomUUID()}`,
    id: idNorm,
    params,
  };
}

export function parseIndicatorSelections(query: string): IndicatorSelection[] {
  const parts = query.split(",").filter(Boolean);
  if (parts.length === 0) {
    return defaultIndicatorSelections();
  }
  return parts.map((part, index) => {
    const [id, paramsStr] = part.includes(":")
      ? part.split(":", 2)
      : [part.trim(), "20"];
    const idNorm = id.trim().toLowerCase();
    const params = parseParams(idNorm, paramsStr);
    return {
      slotId: `slot-${index}-${idNorm}-${paramsSuffix(idNorm, params)}`,
      id: idNorm,
      params,
    };
  });
}

export function defaultIndicatorSelections(): IndicatorSelection[] {
  return [
    createSelection("sma", { period: 5 }),
    createSelection("sma", { period: 20 }),
  ];
}

export const INDICATOR_COLORS = [
  "#2962FF",
  "#E91E63",
  "#FF9800",
  "#9C27B0",
  "#00BCD4",
  "#4CAF50",
];

/** MACD main line — kept separate from signal so slot palette cannot collide. */
export const MACD_LINE_COLOR = "#2962FF";
export const MACD_SIGNAL_COLOR = "#FF9800";
export const BBANDS_MIDDLE_COLOR = "#94a3b8";

export function colorForSlot(index: number): string {
  return INDICATOR_COLORS[index % INDICATOR_COLORS.length];
}

export function buildColorMap(
  selections: IndicatorSelection[],
): Record<string, string> {
  const map: Record<string, string> = {};
  selections.forEach((sel, index) => {
    const color = colorForSlot(index);
    if (sel.id === "macd") {
      const suffix = paramsSuffix(sel.id, sel.params);
      map[`macd_${suffix}`] = MACD_LINE_COLOR;
      map[`macd_signal_${suffix}`] = MACD_SIGNAL_COLOR;
      return;
    }
    if (sel.id === "bbands") {
      const suffix = paramsSuffix(sel.id, sel.params);
      map[`bbands_upper_${suffix}`] = color;
      map[`bbands_middle_${suffix}`] = BBANDS_MIDDLE_COLOR;
      map[`bbands_lower_${suffix}`] = color;
      return;
    }
    indicatorSeriesKeys(sel).forEach((key) => {
      map[key] = color;
    });
  });
  return map;
}

export function removeSelection(
  selections: IndicatorSelection[],
  slotId: string,
): IndicatorSelection[] {
  return selections.filter((s) => s.slotId !== slotId);
}

export function updateSelectionParams(
  selections: IndicatorSelection[],
  slotId: string,
  params: Record<string, number>,
): IndicatorSelection[] {
  const current = selections.find((s) => s.slotId === slotId);
  if (!current) return selections;
  if (
    selections.some(
      (s) =>
        s.slotId !== slotId &&
        s.id === current.id &&
        paramsEqual(s.params, params),
    )
  ) {
    return selections;
  }
  return selections.map((s) =>
    s.slotId === slotId ? { ...s, params } : s,
  );
}

export function addSelection(
  selections: IndicatorSelection[],
  id: string,
  params: Record<string, number>,
): IndicatorSelection[] {
  if (hasDuplicate(selections, id, params)) {
    return selections;
  }
  return [...selections, createSelection(id, params)];
}

export function paramsDisplay(params: Record<string, number>): string {
  return orderedParamEntries("", params)
    .map(([, value]) => String(value))
    .join("/");
}

function selectionQueryPart(selection: IndicatorSelection): string {
  const entries = orderedParamEntries(selection.id, selection.params);
  if (entries.length === 1 && entries[0][0] === "period") {
    return `${selection.id}:${entries[0][1]}`;
  }
  return `${selection.id}:${entries
    .map(([key, value]) => `${key}=${value}`)
    .join(";")}`;
}

function parseParams(id: string, raw: string): Record<string, number> {
  if (!raw.includes("=")) {
    return { period: parseNumber(raw, defaultParams(id).period ?? 20) };
  }

  const parsed: Record<string, number> = {};
  raw.split(";").forEach((pair) => {
    const [key, value] = pair.split("=", 2);
    const trimmedKey = key?.trim();
    if (!trimmedKey) return;
    parsed[trimmedKey] = parseNumber(value, defaultParams(id)[trimmedKey] ?? 1);
  });
  return { ...defaultParams(id), ...parsed };
}

function defaultParams(id: string): Record<string, number> {
  if (id === "macd") return { fast: 12, slow: 26, signal: 9 };
  if (id === "bbands") return { period: 20, std: 2 };
  if (id === "rsi") return { period: 14 };
  return { period: 20 };
}

function parseNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function orderedParamEntries(
  id: string,
  params: Record<string, number>,
): [string, number][] {
  const order =
    id === "macd"
      ? ["fast", "slow", "signal"]
      : id === "bbands"
        ? ["period", "std"]
        : ["period"];
  const known = order
    .filter((key) => params[key] != null)
    .map((key) => [key, params[key]] as [string, number]);
  const extra = Object.entries(params).filter(([key]) => !order.includes(key));
  return [...known, ...extra];
}

function paramsSuffix(id: string, params: Record<string, number>): string {
  return orderedParamEntries(id, params)
    .map(([, value]) => String(value))
    .join("_");
}

function paramsEqual(
  left: Record<string, number>,
  right: Record<string, number>,
): boolean {
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key, index) => key === rightKeys[index] && left[key] === right[key])
  );
}

function indicatorSeriesKeys(selection: IndicatorSelection): string[] {
  const key = selectionApiKey(selection);
  if (selection.id === "macd") {
    const suffix = paramsSuffix(selection.id, selection.params);
    return [`macd_${suffix}`, `macd_signal_${suffix}`, `macd_hist_${suffix}`];
  }
  if (selection.id === "bbands") {
    const suffix = paramsSuffix(selection.id, selection.params);
    return [
      `bbands_upper_${suffix}`,
      `bbands_middle_${suffix}`,
      `bbands_lower_${suffix}`,
    ];
  }
  return [key];
}
