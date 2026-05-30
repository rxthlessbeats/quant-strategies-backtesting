import { format } from "date-fns";

const DAILY_MARKET_AS_OF_HOUR = 17;

export function dailyMarketAsOfDate(timestamp: number | null | undefined): Date | null {
  if (timestamp == null) return null;

  const tradingDay = new Date(timestamp * 1000);
  if (Number.isNaN(tradingDay.getTime())) return null;

  return new Date(
    tradingDay.getUTCFullYear(),
    tradingDay.getUTCMonth(),
    tradingDay.getUTCDate(),
    DAILY_MARKET_AS_OF_HOUR,
    0,
    0,
    0,
  );
}

export function formatDailyMarketAsOf(
  timestamp: number | null | undefined,
): string | null {
  const asOf = dailyMarketAsOfDate(timestamp);
  return asOf ? format(asOf, "M/d/yyyy h:mm a") : null;
}
