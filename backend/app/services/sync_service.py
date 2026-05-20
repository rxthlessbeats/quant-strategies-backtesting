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


def sync_symbol(
    db: Session,
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> DataSource:
    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)

    cached = crud.get_bars(db, symbol, interval, start_ts, end_ts)
    fresh = crud.is_fresh(db, symbol, interval) if interval == "1d" else False
    has_coverage = len(cached) > 0
    covers_start = has_coverage and cached[0].ts <= start_ts

    if has_coverage and fresh and covers_start:
        return DataSource.CACHE

    fetch_start = start_date
    meta = crud.get_fetch_meta(db, symbol, interval)
    if has_coverage and not covers_start:
        fetch_start = start_date
    elif meta and meta.last_bar_ts and not fresh:
        last_dt = pd.to_datetime(meta.last_bar_ts, unit="s", utc=True)
        fetch_start = _next_day(last_dt.strftime("%Y-%m-%d"))

    downloader = get_downloader()
    df = downloader.yahoo(symbol, fetch_start, end_date, interval)
    if df is None or df.empty:
        return DataSource.CACHE if has_coverage else DataSource.FETCH

    rows = crud.save_bars_from_dataframe(db, df, symbol, interval)
    last_ts = max(row.ts for row in rows) if rows else None
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
