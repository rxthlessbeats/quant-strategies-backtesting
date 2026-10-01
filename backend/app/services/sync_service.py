import pandas as pd
from sqlalchemy.orm import Session

from app.db import crud
from app.db.locks import symbol_lock
from app.fetch.downloader import get_downloader
from app.schemas.common import DataSource
from app.schemas.db import FetchMetaRow


def _to_ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str).timestamp())


def _today() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


def sync_symbol(
    db: Session,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
) -> DataSource:
    # Fundamentals requests must not hold up price history for the same ticker.
    with symbol_lock(f"bars:{symbol}:{interval}"):
        db.expire_all()
        return _sync_symbol(db, symbol, start_date, end_date, interval)


def _sync_symbol(
    db: Session,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
) -> DataSource:
    bounded = start_date is not None and end_date is not None
    first_ts, _ = crud.get_bar_bounds(
        db,
        symbol,
        interval,
        _to_ts(start_date) if start_date else None,
        _to_ts(end_date) if end_date else None,
    )
    had_coverage = first_ts is not None
    meta = crud.get_fetch_meta(db, symbol, interval)
    fresh = crud.is_fresh(db, symbol, interval)
    expected_end = pd.to_datetime(
        crud.last_expected_daily_ts(), unit="s", utc=True
    ).strftime("%Y-%m-%d")
    full_history = bool(meta and meta.start_date is None and meta.end_date)
    covers_start = bool(
        meta
        and start_date
        and (full_history or (meta.start_date and meta.start_date <= start_date))
    )
    covers_end = bool(
        meta
        and meta.end_date
        and end_date
        and meta.end_date >= min(end_date, expected_end)
    )
    historical = bool(interval == "1d" and end_date and end_date < expected_end)

    if bounded and covers_start and covers_end and (fresh or historical):
        return DataSource.CACHE
    if not bounded and full_history and fresh:
        return DataSource.CACHE

    fetch_start = start_date
    fetch_end = end_date or _today()
    if meta and meta.last_bar_ts and (full_history or (bounded and covers_start)):
        # Include the latest session again: its daily bar may still be changing.
        last_day = pd.to_datetime(meta.last_bar_ts, unit="s", utc=True).strftime(
            "%Y-%m-%d"
        )
        refresh_start = min(last_day, meta.end_date or last_day)
        if refresh_start <= fetch_end:
            fetch_start = refresh_start

    downloader = get_downloader()
    df = (
        downloader.yahoo(symbol, fetch_start, fetch_end, interval)
        if fetch_start
        else downloader.yahoo_max(symbol, interval)
    )
    if df is not None and not df.empty:
        crud.save_bars_from_dataframe(db, df, symbol, interval)

    _, last_ts = crud.get_bar_bounds(db, symbol, interval)
    # A null start marks a successful request for all available history.
    coverage_start = start_date if bounded else None
    coverage_end = min(fetch_end, _today())
    if bounded and full_history:
        coverage_start = None
        coverage_end = max(coverage_end, meta.end_date)
    elif meta and meta.start_date and meta.end_date and coverage_start:
        # Merge only overlapping/adjacent fetched ranges; disjoint requests may have gaps.
        day = pd.Timedelta(days=1)
        overlaps_meta_end = (
            pd.Timestamp(coverage_start) <= pd.Timestamp(meta.end_date) + day
        )
        overlaps_meta_start = (
            pd.Timestamp(meta.start_date) <= pd.Timestamp(coverage_end) + day
        )
        if overlaps_meta_end and overlaps_meta_start:
            coverage_start = min(coverage_start, meta.start_date)
            coverage_end = max(coverage_end, meta.end_date)
    crud.upsert_fetch_meta(
        db,
        FetchMetaRow(
            symbol=symbol,
            interval=interval,
            last_bar_ts=last_ts,
            start_date=coverage_start,
            end_date=coverage_end,
        ),
    )
    return DataSource.CACHE if had_coverage else DataSource.FETCH
