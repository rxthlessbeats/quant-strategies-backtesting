from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.models import Bar, CompanyFundamentals, FetchMeta, MarketDataModule
from app.schemas.converters import bar_rows_from_dataframe, bars_to_dataframe
from app.schemas.db import (
    BarRow,
    CompanyFundamentalsRow,
    FetchMetaRow,
    MarketDataModuleRow,
)


def _is_sqlite(session: Session) -> bool:
    return session.bind.dialect.name == "sqlite"


def get_bars(
    db: Session,
    symbol: str,
    interval: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[Bar]:
    conditions = [
        Bar.symbol == symbol,
        Bar.interval == interval,
    ]
    if start_ts is not None:
        conditions.append(Bar.ts >= start_ts)
    if end_ts is not None:
        conditions.append(Bar.ts <= end_ts)

    stmt = select(Bar).where(*conditions).order_by(Bar.ts)
    return list(db.scalars(stmt).all())


def upsert_bars(db: Session, rows: list[BarRow]) -> int:
    if not rows:
        return 0

    payloads = [row.to_orm_dict() for row in rows]

    if _is_sqlite(db):
        stmt = sqlite_insert(Bar).values(payloads)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "ts"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "adj_close": stmt.excluded.adj_close,
            },
        )
        db.execute(stmt)
    else:
        for payload in payloads:
            existing = db.scalar(
                select(Bar).where(
                    Bar.symbol == payload["symbol"],
                    Bar.interval == payload["interval"],
                    Bar.ts == payload["ts"],
                )
            )
            if existing:
                for key in ("open", "high", "low", "close", "volume", "adj_close"):
                    setattr(existing, key, payload[key])
            else:
                db.add(Bar(**payload))

    db.commit()
    return len(rows)


def get_fetch_meta(db: Session, symbol: str, interval: str) -> FetchMeta | None:
    return db.get(FetchMeta, (symbol, interval))


def fetch_meta_to_schema(meta: FetchMeta | None) -> FetchMetaRow | None:
    if meta is None:
        return None
    return FetchMetaRow(
        symbol=meta.symbol,
        interval=meta.interval,
        last_bar_ts=meta.last_bar_ts,
        fetched_at=meta.fetched_at,
        start_date=meta.start_date,
        end_date=meta.end_date,
    )


def upsert_fetch_meta(db: Session, row: FetchMetaRow) -> FetchMeta:
    now = datetime.now(timezone.utc).isoformat()
    meta = get_fetch_meta(db, row.symbol, row.interval)
    if meta is None:
        meta = FetchMeta(
            symbol=row.symbol,
            interval=row.interval,
            last_bar_ts=row.last_bar_ts,
            fetched_at=now,
            start_date=row.start_date,
            end_date=row.end_date,
        )
        db.add(meta)
    else:
        meta.last_bar_ts = row.last_bar_ts
        meta.fetched_at = now
        meta.start_date = row.start_date
        meta.end_date = row.end_date
    db.commit()
    db.refresh(meta)
    return meta


def get_market_data_module(
    db: Session, symbol: str, module: str
) -> MarketDataModule | None:
    stmt = select(MarketDataModule).where(
        MarketDataModule.symbol == symbol.upper(),
        MarketDataModule.module == module,
    )
    return db.scalar(stmt)


def get_market_data_modules(
    db: Session, symbol: str, modules: list[str] | None = None
) -> list[MarketDataModule]:
    conditions = [MarketDataModule.symbol == symbol.upper()]
    if modules:
        conditions.append(MarketDataModule.module.in_(modules))
    stmt = select(MarketDataModule).where(*conditions).order_by(MarketDataModule.module)
    return list(db.scalars(stmt).all())


def market_data_module_to_schema(
    row: MarketDataModule | None,
) -> MarketDataModuleRow | None:
    if row is None:
        return None
    return MarketDataModuleRow(
        **{
            column.name: getattr(row, column.name)
            for column in MarketDataModule.__table__.columns
            if column.name != "id"
        }
    )


def upsert_market_data_module(
    db: Session, row: MarketDataModuleRow
) -> MarketDataModule:
    payload = row.to_orm_dict()
    payload["symbol"] = row.symbol.upper()
    if _is_sqlite(db):
        stmt = sqlite_insert(MarketDataModule).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "module"],
            set_={
                "payload_json": stmt.excluded.payload_json,
                "payload_hash": stmt.excluded.payload_hash,
                "last_checked_at": stmt.excluded.last_checked_at,
                "last_changed_at": stmt.excluded.last_changed_at,
                "fetched_at": stmt.excluded.fetched_at,
                "next_refresh_at": stmt.excluded.next_refresh_at,
                "latest_event_date": stmt.excluded.latest_event_date,
                "source": stmt.excluded.source,
                "status": stmt.excluded.status,
            },
        )
        db.execute(stmt)
        db.commit()
        saved = get_market_data_module(db, row.symbol, row.module)
        if saved is None:
            raise ValueError("Market data module upsert failed")
        return saved

    existing = get_market_data_module(db, row.symbol, row.module)
    if existing is None:
        existing = MarketDataModule(**payload)
        db.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)

    db.commit()
    db.refresh(existing)
    return existing


def get_company_fundamentals(db: Session, symbol: str) -> CompanyFundamentals | None:
    return db.get(CompanyFundamentals, symbol.upper())


def company_fundamentals_to_schema(
    fundamentals: CompanyFundamentals | None,
) -> CompanyFundamentalsRow | None:
    if fundamentals is None:
        return None
    return CompanyFundamentalsRow(
        **{
            column.name: getattr(fundamentals, column.name)
            for column in CompanyFundamentals.__table__.columns
        }
    )


def upsert_company_fundamentals(
    db: Session, row: CompanyFundamentalsRow
) -> CompanyFundamentals:
    payload = row.to_orm_dict()
    payload["symbol"] = row.symbol.upper()
    now = datetime.now(timezone.utc).isoformat()
    payload["fetched_at"] = now

    fundamentals = get_company_fundamentals(db, row.symbol)
    if fundamentals is None:
        fundamentals = CompanyFundamentals(**payload)
        db.add(fundamentals)
    else:
        for key, value in payload.items():
            setattr(fundamentals, key, value)

    db.commit()
    db.refresh(fundamentals)
    return fundamentals


def last_expected_daily_ts() -> int:
    today = pd.Timestamp.now(tz="UTC").normalize()
    prev_bday = today - pd.offsets.BDay(1)
    return int(prev_bday.timestamp())


def _last_expected_daily_date() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize() - pd.offsets.BDay(1)


def is_fresh(db: Session, symbol: str, interval: str) -> bool:
    if interval != "1d":
        return False
    meta = get_fetch_meta(db, symbol, interval)
    if meta is None or meta.last_bar_ts is None:
        return False
    last_bar_day = pd.to_datetime(meta.last_bar_ts, unit="s", utc=True).normalize()
    return last_bar_day >= _last_expected_daily_date()


def load_bars_dataframe(
    db: Session,
    symbol: str,
    interval: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    return bars_to_dataframe(get_bars(db, symbol, interval, start_ts, end_ts))


def save_bars_from_dataframe(
    db: Session, df: pd.DataFrame, symbol: str, interval: str
) -> list[BarRow]:
    rows = bar_rows_from_dataframe(df, symbol, interval)
    upsert_bars(db, rows)
    return rows
