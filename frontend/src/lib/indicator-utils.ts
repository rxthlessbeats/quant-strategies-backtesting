export interface IndicatorSelection {
  slotId: string;
  id: string;
  period: number;
}

export function buildIndicatorsQuery(
  selections: IndicatorSelection[],
): string {
  return selections.map((s) => `${s.id}:${s.period}`).join(",");
}

export function selectionApiKey(sel: IndicatorSelection): string {
  return `${sel.id}_${sel.period}`;
}

export function hasDuplicate(
  selections: IndicatorSelection[],
  id: string,
  period: number,
): boolean {
  return selections.some((s) => s.id === id && s.period === period);
}

export function createSelection(id: string, period: number): IndicatorSelection {
  return {
    slotId: `${id}-${period}-${crypto.randomUUID()}`,
    id: id.toLowerCase(),
    period,
  };
}

export function parseIndicatorSelections(query: string): IndicatorSelection[] {
  const parts = query.split(",").filter(Boolean);
  if (parts.length === 0) {
    return defaultIndicatorSelections();
  }
  return parts.map((part, index) => {
    const [id, periodStr] = part.includes(":")
      ? part.split(":", 2)
      : [part.trim(), "20"];
    const idNorm = id.trim().toLowerCase();
    const period = parseInt(periodStr, 10) || 20;
    return {
      slotId: `slot-${index}-${idNorm}-${period}`,
      id: idNorm,
      period,
    };
  });
}

export function defaultIndicatorSelections(): IndicatorSelection[] {
  return [
    createSelection("sma", 5),
    createSelection("sma", 20),
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

export function colorForSlot(index: number): string {
  return INDICATOR_COLORS[index % INDICATOR_COLORS.length];
}

export function buildColorMap(
  selections: IndicatorSelection[],
): Record<string, string> {
  const map: Record<string, string> = {};
  selections.forEach((sel, index) => {
    map[selectionApiKey(sel)] = colorForSlot(index);
  });
  return map;
}

export function removeSelection(
  selections: IndicatorSelection[],
  slotId: string,
): IndicatorSelection[] {
  return selections.filter((s) => s.slotId !== slotId);
}

export function updateSelectionPeriod(
  selections: IndicatorSelection[],
  slotId: string,
  period: number,
): IndicatorSelection[] {
  const next = Math.max(1, period);
  const current = selections.find((s) => s.slotId === slotId);
  if (!current) return selections;
  if (
    selections.some(
      (s) => s.slotId !== slotId && s.id === current.id && s.period === next,
    )
  ) {
    return selections;
  }
  return selections.map((s) =>
    s.slotId === slotId ? { ...s, period: next } : s,
  );
}

export function addSelection(
  selections: IndicatorSelection[],
  id: string,
  period: number,
): IndicatorSelection[] {
  if (hasDuplicate(selections, id, period)) {
    return selections;
  }
  return [...selections, createSelection(id, period)];
}
