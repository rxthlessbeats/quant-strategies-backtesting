export function dailyMarketAsOfDate(timestamp: number | null | undefined): Date | null {
  if (timestamp == null) return null;
  const date = new Date(timestamp * 1000);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatDailyMarketAsOf(timestamp: number | null | undefined): string | null {
  const date = dailyMarketAsOfDate(timestamp);
  return date ? date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }) : null;
}
