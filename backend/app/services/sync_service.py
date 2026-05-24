import pandas as pd
from sqlalchemy.orm import Session

from app.db import crud
from app.fetch.downloader import get_downloader
from app.schemas.common import DataSource
from app.schemas.db import FetchMetaRow


def _to_ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str).timestamp())


def _next_day(date_str: str) -> str:
    d = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _today() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


def sync_symbol(
    db: Session,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
) -> DataSource:
    bounded = start_date is not None and end_date is not None
    start_ts = _to_ts(start_date) if start_date else None
    end_ts = _to_ts(end_date) if end_date else None

    cached = crud.get_bars(db, symbol, interval, start_ts, end_ts)
    fresh = crud.is_fresh(db, symbol, interval) if interval == "1d" else False
    has_coverage = len(cached) > 0
    covers_start = bool(start_ts is None or (has_coverage and cached[0].ts <= start_ts))

    downloader = get_downloader()

    if not bounded:
        meta = crud.get_fetch_meta(db, symbol, interval)
        if has_coverage and fresh:
            return DataSource.CACHE

        if meta and meta.last_bar_ts:
            last_dt = pd.to_datetime(meta.last_bar_ts, unit="s", utc=True)
            fetch_start = _next_day(last_dt.strftime("%Y-%m-%d"))
            fetch_end = _today()
            df = downloader.yahoo(symbol, fetch_start, fetch_end, interval)
        else:
            df = downloader.yahoo_max(symbol, interval)

        if df is None or df.empty:
            return DataSource.CACHE if has_coverage else DataSource.FETCH

        crud.save_bars_from_dataframe(db, df, symbol, interval)
        all_bars = crud.get_bars(db, symbol, interval)
        last_ts = max(row.ts for row in all_bars) if all_bars else None
        first_ts = min(row.ts for row in all_bars) if all_bars else None
        crud.upsert_fetch_meta(
            db,
            FetchMetaRow(
                symbol=symbol,
                interval=interval,
                last_bar_ts=last_ts,
                start_date=(
                    pd.to_datetime(first_ts, unit="s", utc=True).strftime("%Y-%m-%d")
                    if first_ts
                    else None
                ),
                end_date=(
                    pd.to_datetime(last_ts, unit="s", utc=True).strftime("%Y-%m-%d")
                    if last_ts
                    else None
                ),
            ),
        )
        return DataSource.FETCH

    if has_coverage and fresh and covers_start:
        return DataSource.CACHE

    assert start_date is not None
    assert end_date is not None
    fetch_start = start_date
    meta = crud.get_fetch_meta(db, symbol, interval)
    if has_coverage and not covers_start:
        fetch_start = start_date
    elif meta and meta.last_bar_ts and not fresh:
        last_dt = pd.to_datetime(meta.last_bar_ts, unit="s", utc=True)
        fetch_start = _next_day(last_dt.strftime("%Y-%m-%d"))

    df = downloader.yahoo(symbol, fetch_start, end_date, interval)
    if df is None or df.empty:
        return DataSource.CACHE if has_coverage else DataSource.FETCH

    crud.save_bars_from_dataframe(db, df, symbol, interval)
    all_bars = crud.get_bars(db, symbol, interval)
    last_ts = max(row.ts for row in all_bars) if all_bars else None
    crud.upsert_fetch_meta(
        db,
        FetchMetaRow(
            symbol=symbol,
            interval=interval,
            last_bar_ts=last_ts,
            start_date=start_date,
            end_date=end_date,
        ),
    )
    return DataSource.FETCH
