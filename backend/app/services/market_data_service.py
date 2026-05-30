import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import sqlite_write
from app.fetch.yahoo import DataDownloader
from app.schemas.db import MarketDataModuleRow
from app.schemas.settings import settings

OVERVIEW_MODULES = [
    "assetProfile",
    "summaryProfile",
    "defaultKeyStatistics",
    "summaryDetail",
    "calendarEvents",
    "financialData",
    "recommendationTrend",
    "price",
]

AREA_MODULES: dict[str, list[str]] = {
    "profile": ["assetProfile", "summaryProfile", "price"],
    "statistics": ["summaryDetail", "defaultKeyStatistics", "financialData"],
    "financial-snapshot": ["calendarEvents", "financialData"],
    "earnings": ["calendarEvents", "earnings", "earningsTrend", "earningsHistory"],
    "statements": [
        "incomeStatementHistory",
        "incomeStatementHistoryQuarterly",
        "balanceSheetHistoryQuarterly",
        "cashflowStatementHistoryQuarterly",
        "fundamentalsTimeSeriesQuarterly",
    ],
    "analysts": ["recommendationTrend", "upgradeDowngradeHistory", "insights"],
    "ownership": [
        "institutionOwnership",
        "fundOwnership",
        "majorHoldersBreakdown",
    ],
    "insiders": ["insiderTransactions"],
    "filings": ["secFilings"],
}

DAILY_MODULES = {"recommendationTrend", "upgradeDowngradeHistory"}
WEEKLY_MODULES = {
    "summaryDetail",
    "defaultKeyStatistics",
    "calendarEvents",
    "earningsTrend",
    "insiderTransactions",
    "secFilings",
}
MONTHLY_MODULES = {"assetProfile", "summaryProfile", "price"}
QUARTERLY_MODULES = {
    "institutionOwnership",
    "fundOwnership",
    "majorHoldersBreakdown",
}
EVENT_DRIVEN_EARNINGS_MODULES = {
    "earnings",
    "earningsHistory",
    "incomeStatementHistory",
    "incomeStatementHistoryQuarterly",
    "balanceSheetHistoryQuarterly",
    "cashflowStatementHistoryQuarterly",
    "fundamentalsTimeSeriesQuarterly",
}


def modules_for_area(area: str) -> list[str]:
    try:
        return AREA_MODULES[area]
    except KeyError as exc:
        raise ValueError(f"Unsupported market data area: {area}") from exc


def ensure_area(db: Session, symbol: str, area: str, force: bool = False) -> list[dict]:
    return ensure_modules(db, symbol, modules_for_area(area), force)


def ensure_modules(
    db: Session, symbol: str, modules: list[str], force: bool = False
) -> list[dict]:
    normalized = symbol.upper()
    requested = _dedupe(modules)
    cached = {
        row.module: row
        for row in crud.get_market_data_modules(db, normalized, requested)
    }
    due = [
        module for module in requested if force or _is_module_due(cached.get(module))
    ]

    if due:
        downloader = DataDownloader()
        timeseries_due = [
            module for module in due if module == "fundamentalsTimeSeriesQuarterly"
        ]
        insights_due = [module for module in due if module == "insights"]
        special_due = set(timeseries_due + insights_due)
        quote_due = [module for module in due if module not in special_due]
        payload = downloader.quote_summary(normalized, quote_due) if quote_due else {}
        if timeseries_due:
            payload["fundamentalsTimeSeriesQuarterly"] = (
                downloader.fundamentals_timeseries(normalized)
            )
        if insights_due:
            payload["insights"] = downloader.insights(normalized)
        now = datetime.now(timezone.utc)

        def persist_modules() -> None:
            for module in due:
                module_payload = payload.get(module)
                if module_payload is None:
                    continue
                previous = cached.get(module)
                saved = _save_module(
                    db,
                    normalized,
                    module,
                    module_payload,
                    previous,
                    now,
                    commit=False,
                )
                cached[module] = saved
            crud.commit_session(db)

        with sqlite_write():
            try:
                persist_modules()
            except Exception:
                db.rollback()
                raise

    return [
        _module_response(crud.market_data_module_to_schema(cached[module]))
        for module in requested
        if module in cached
    ]


def get_cached_modules(
    db: Session, symbol: str, modules: list[str] | None = None
) -> list[dict]:
    rows = crud.get_market_data_modules(db, symbol.upper(), modules)
    return [_module_response(crud.market_data_module_to_schema(row)) for row in rows]


def get_company_overview_payload(db: Session, symbol: str) -> dict[str, Any]:
    modules = ensure_modules(db, symbol, OVERVIEW_MODULES)
    module_payloads = {item["module"]: item["payload"] for item in modules}
    return _overview_from_modules(symbol.upper(), module_payloads)


def _save_module(
    db: Session,
    symbol: str,
    module: str,
    payload: dict,
    previous,
    now: datetime,
    *,
    commit: bool = True,
):
    payload_hash = _payload_hash(payload)
    previous_hash = previous.payload_hash if previous is not None else None
    changed = previous is None or previous_hash != payload_hash
    changed_at = now.isoformat() if changed else previous.last_changed_at
    row = MarketDataModuleRow(
        symbol=symbol,
        module=module,
        payload_json=payload if changed else previous.payload_json,
        payload_hash=payload_hash,
        last_checked_at=now.isoformat(),
        last_changed_at=changed_at,
        fetched_at=now.isoformat() if changed else previous.fetched_at,
        next_refresh_at=_next_refresh_at(db, symbol, module, payload, now).isoformat(),
        latest_event_date=_latest_event_date(module, payload),
        source="yahoo",
        status="ok",
    )
    return crud.upsert_market_data_module(db, row, commit=commit)


def _is_module_due(row) -> bool:
    if row is None or not row.next_refresh_at:
        return True
    try:
        next_refresh = datetime.fromisoformat(row.next_refresh_at)
    except ValueError:
        return True
    if next_refresh.tzinfo is None:
        next_refresh = next_refresh.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= next_refresh


def _next_refresh_at(
    db: Session, symbol: str, module: str, payload: dict, now: datetime
) -> datetime:
    if module == "financialData":
        earnings_refresh = _next_earnings_refresh(db, symbol, now)
        weekly_refresh = now + timedelta(days=7)
        if earnings_refresh and earnings_refresh < weekly_refresh:
            return earnings_refresh
        return weekly_refresh
    if module in EVENT_DRIVEN_EARNINGS_MODULES:
        return _next_earnings_refresh(db, symbol, now) or now + timedelta(days=90)
    if module in DAILY_MODULES:
        return now + timedelta(days=1)
    if module in WEEKLY_MODULES:
        return now + timedelta(days=7)
    if module in MONTHLY_MODULES:
        return now + timedelta(days=30)
    if module in QUARTERLY_MODULES:
        return now + timedelta(days=90)
    return now + timedelta(days=7)


def _next_earnings_refresh(db: Session, symbol: str, now: datetime) -> datetime | None:
    calendar = crud.get_market_data_module(db, symbol, "calendarEvents")
    if calendar is None:
        return None
    refresh = _extract_earnings_refresh_at(calendar.payload_json)
    if refresh is None:
        return None
    if refresh < now:
        return None
    return refresh


def _extract_earnings_refresh_at(payload: dict) -> datetime | None:
    value = payload.get("earningsDate")
    if isinstance(value, list):
        value = value[0] if value else None
    earnings_at = _parse_earnings_datetime(value)
    if earnings_at is None:
        return None
    market_tz = ZoneInfo(settings.refresh_timezone)
    local_at = earnings_at.astimezone(market_tz)
    fmt = value.get("fmt") if isinstance(value, dict) else None
    has_exact_time = _has_exact_earnings_time(local_at, fmt)
    if has_exact_time:
        refresh_at = local_at + timedelta(hours=settings.earnings_refresh_delay_hours)
    else:
        refresh_at = local_at.replace(
            hour=settings.refresh_market_close_hour,
            minute=settings.refresh_market_close_minute,
            second=0,
            microsecond=0,
        ) + timedelta(hours=settings.earnings_refresh_delay_hours)
    return refresh_at.astimezone(timezone.utc)


def _parse_earnings_datetime(value: Any) -> datetime | None:
    raw = _raw(value)
    if raw is None:
        return None
    try:
        timestamp = int(float(raw))
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        pass
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _has_exact_earnings_time(earnings_at: datetime, fmt: str | None) -> bool:
    if fmt is not None:
        return ":" in fmt or "AM" in fmt.upper() or "PM" in fmt.upper()
    return any([earnings_at.hour, earnings_at.minute, earnings_at.second])


def _latest_event_date(module: str, payload: dict) -> str | None:
    event_keys = {
        "upgradeDowngradeHistory": ("history", "epochGradeDate"),
        "insiderTransactions": ("transactions", "startDate"),
        "secFilings": ("filings", "epochDate"),
        "earningsHistory": ("history", "quarter"),
    }
    list_key, date_key = event_keys.get(module, ("", ""))
    items = payload.get(list_key)
    if not isinstance(items, list):
        return None
    values = [_raw(item.get(date_key)) for item in items if isinstance(item, dict)]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _module_response(row: MarketDataModuleRow | None) -> dict:
    if row is None:
        raise ValueError("Market data module not found")
    return {
        "symbol": row.symbol,
        "module": row.module,
        "payload": row.payload_json,
        "payload_hash": row.payload_hash,
        "last_checked_at": row.last_checked_at,
        "last_changed_at": row.last_changed_at,
        "fetched_at": row.fetched_at,
        "next_refresh_at": row.next_refresh_at,
        "latest_event_date": row.latest_event_date,
        "source": row.source,
        "status": row.status,
    }


def _overview_from_modules(symbol: str, data: dict[str, Any]) -> dict[str, Any]:
    profile = data.get("assetProfile") or data.get("summaryProfile") or {}
    stats = data.get("defaultKeyStatistics") or {}
    financial = data.get("financialData") or {}
    detail = data.get("summaryDetail") or {}
    calendar = data.get("calendarEvents") or {}
    price = data.get("price") or {}
    trend = data.get("recommendationTrend") or {}

    address = ", ".join(
        part
        for part in [
            profile.get("address1"),
            profile.get("city"),
            profile.get("state"),
            profile.get("zip"),
        ]
        if part
    )
    latest_trend = _latest_recommendation_trend(trend)

    return {
        "Symbol": symbol,
        "AssetType": price.get("quoteType") or price.get("typeDisp"),
        "Name": _raw(price.get("longName")) or _raw(price.get("shortName")) or symbol,
        "Description": profile.get("longBusinessSummary"),
        "CIK": None,
        "Exchange": _raw(price.get("exchangeName")) or _raw(price.get("exchange")),
        "Currency": _raw(price.get("currency")),
        "Country": profile.get("country"),
        "Sector": profile.get("sector"),
        "Industry": profile.get("industry"),
        "Address": address or None,
        "FiscalYearEnd": _raw(calendar.get("fiscalYearEnd")),
        "LatestQuarter": _raw(calendar.get("earningsDate")),
        "EBITDA": _raw(financial.get("ebitda")),
        "BookValue": _raw(stats.get("bookValue")),
        "DividendPerShare": _raw(detail.get("dividendRate")),
        "EPS": _raw(stats.get("trailingEps")),
        "RevenuePerShareTTM": _raw(financial.get("revenuePerShare")),
        "ProfitMargin": _raw(financial.get("profitMargins")),
        "OperatingMarginTTM": _raw(financial.get("operatingMargins")),
        "ReturnOnAssetsTTM": _raw(financial.get("returnOnAssets")),
        "ReturnOnEquityTTM": _raw(financial.get("returnOnEquity")),
        "RevenueTTM": _raw(financial.get("totalRevenue")),
        "GrossProfitTTM": _raw(financial.get("grossProfits")),
        "DilutedEPSTTM": _raw(stats.get("trailingEps")),
        "QuarterlyEarningsGrowthYOY": _raw(financial.get("earningsGrowth")),
        "QuarterlyRevenueGrowthYOY": _raw(financial.get("revenueGrowth")),
        "SharesOutstanding": _raw(stats.get("sharesOutstanding")),
        "DividendDate": _raw(calendar.get("dividendDate")),
        "ExDividendDate": _raw(calendar.get("exDividendDate")),
        "AnalystRatingStrongBuy": _raw(latest_trend.get("strongBuy")),
        "AnalystRatingBuy": _raw(latest_trend.get("buy")),
        "AnalystRatingHold": _raw(latest_trend.get("hold")),
        "AnalystRatingSell": _raw(latest_trend.get("sell")),
        "AnalystRatingStrongSell": _raw(latest_trend.get("strongSell")),
    }


def _latest_recommendation_trend(payload: dict) -> dict:
    trend = payload.get("trend")
    if isinstance(trend, list) and trend:
        return trend[0] if isinstance(trend[0], dict) else {}
    return {}


def _payload_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _raw(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("raw") if "raw" in value else value.get("fmt")
    if value is None:
        return None
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
