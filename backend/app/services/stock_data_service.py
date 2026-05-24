import pandas as pd
from sqlalchemy.orm import Session

from app.db import crud
from app.schemas.converters import bar_points_from_dataframe
from app.schemas.market import ChartMeta, OhlcvResult
from app.schemas.requests import ChartQuery
from app.services.sync_service import sync_symbol


def get_ohlcv(db: Session, query: ChartQuery) -> OhlcvResult:
    source = sync_symbol(db, query.symbol, query.start, query.end, query.interval)
    start_ts = int(pd.Timestamp(query.start).timestamp()) if query.start else None
    end_ts = int(pd.Timestamp(query.end).timestamp()) if query.end else None
    df = crud.load_bars_dataframe(db, query.symbol, query.interval, start_ts, end_ts)
    if not df.empty:
        df = df.dropna(subset=["Close"])

    bars = bar_points_from_dataframe(df)
    meta_row = crud.get_fetch_meta(db, query.symbol, query.interval)
    cached_through = None
    if meta_row and meta_row.last_bar_ts:
        cached_through = pd.to_datetime(
            meta_row.last_bar_ts, unit="s", utc=True
        ).strftime("%Y-%m-%d")

    return OhlcvResult(
        symbol=query.symbol,
        interval=query.interval,
        start=(df.index.min().strftime("%Y-%m-%d") if not df.empty else query.start),
        end=(df.index.max().strftime("%Y-%m-%d") if not df.empty else query.end),
        meta=ChartMeta(
            source=source,
            cached_through=cached_through,
            fetched_at=meta_row.fetched_at if meta_row else None,
            bar_count=len(bars),
        ),
        bars=bars,
    )
