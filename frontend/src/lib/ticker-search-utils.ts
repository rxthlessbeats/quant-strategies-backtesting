import type { TickerSearchItem } from "@/lib/types";

const US_EXCHANGE_CODES = new Set([
  "NMS",
  "NYQ",
  "NGM",
  "NCM",
  "ASE",
  "PCX",
  "BTS",
  "PNK",
]);

const US_EXCHANGE_LABELS = [
  "NASDAQ",
  "NYSE",
  "NYSE ARCA",
  "AMEX",
  "NYSEAMERICAN",
  "BATS",
];

export function isUsEquity(item: TickerSearchItem): boolean {
  if (item.type && item.type !== "EQUITY") {
    return false;
  }
  if (item.symbol.includes(".")) {
    return false;
  }
  const region = (item.region ?? "").toUpperCase();
  if (US_EXCHANGE_CODES.has(region)) {
    return true;
  }
  return US_EXCHANGE_LABELS.some((label) => region.includes(label));
}

export function filterUsEquities(items: TickerSearchItem[]): TickerSearchItem[] {
  return items.filter(isUsEquity).slice(0, 8);
}
