from collections.abc import Iterable

from app.db.database import SessionLocal
from app.schemas.settings import settings
from app.services.market_data_service import ensure_modules

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover - dependency is optional until installed
    BackgroundScheduler = None
    CronTrigger = None

ANALYST_MODULES = ["recommendationTrend", "upgradeDowngradeHistory"]
WEEKLY_MODULES = [
    "summaryDetail",
    "defaultKeyStatistics",
    "calendarEvents",
    "earningsTrend",
    "insiderTransactions",
    "secFilings",
    "financialData",
]
MONTHLY_MODULES = [
    "assetProfile",
    "summaryProfile",
    "price",
    "institutionOwnership",
    "fundOwnership",
    "majorHoldersBreakdown",
]
EARNINGS_MODULES = [
    "financialData",
    "earnings",
    "earningsHistory",
    "incomeStatementHistory",
    "incomeStatementHistoryQuarterly",
    "balanceSheetHistoryQuarterly",
    "cashflowStatementHistoryQuarterly",
    "fundamentalsTimeSeriesQuarterly",
]

_scheduler = None


def start_market_data_scheduler() -> None:
    global _scheduler
    if not settings.refresh_scheduler_enabled or not settings.refresh_symbols:
        return
    if BackgroundScheduler is None or CronTrigger is None:
        return
    if _scheduler is not None and _scheduler.running:
        return

    scheduler = BackgroundScheduler(timezone=settings.refresh_timezone)
    scheduler.add_job(
        _refresh_modules,
        CronTrigger(
            day_of_week="mon-fri",
            hour=settings.refresh_market_close_hour,
            minute=settings.refresh_market_close_minute,
        ),
        args=[ANALYST_MODULES],
        id="market-data-analysts-daily",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _refresh_modules,
        CronTrigger(
            day_of_week="sat",
            hour=settings.refresh_market_close_hour,
            minute=settings.refresh_market_close_minute,
        ),
        args=[WEEKLY_MODULES],
        id="market-data-weekly",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _refresh_modules,
        CronTrigger(
            day=1,
            hour=settings.refresh_market_close_hour,
            minute=settings.refresh_market_close_minute,
        ),
        args=[MONTHLY_MODULES],
        id="market-data-monthly",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _refresh_modules,
        CronTrigger(
            day_of_week="mon-fri",
            hour=settings.refresh_market_close_hour,
            minute=settings.refresh_market_close_minute,
        ),
        args=[EARNINGS_MODULES],
        id="market-data-earnings-window",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler


def stop_market_data_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None


def _refresh_modules(modules: Iterable[str]) -> None:
    module_list = list(modules)
    for symbol in settings.refresh_symbols:
        db = SessionLocal()
        try:
            ensure_modules(db, symbol, module_list)
        finally:
            db.close()
