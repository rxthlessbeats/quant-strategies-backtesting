import math

import pandas as pd

from app.db.models import Bar
from app.schemas.common import BarPoint
from app.schemas.db import BarRow


def bar_points_from_dataframe(df: pd.DataFrame) -> list[BarPoint]:
    bars: list[BarPoint] = []
    for ts, (open_, high, low, close, volume, adj) in _bar_values(df):
        if pd.isna(close):
            continue
        bars.append(
            BarPoint(
                timestamp=int(ts),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume) if pd.notna(volume) else 0.0,
                adj_close=float(adj) if pd.notna(adj) else float(close),
            )
        )
    return bars


def bar_rows_from_dataframe(
    df: pd.DataFrame, symbol: str, interval: str
) -> list[BarRow]:
    rows: list[BarRow] = []
    for ts, (open_, high, low, close, volume, adj) in _bar_values(df):
        if pd.isna(close):
            continue
        rows.append(
            BarRow(
                symbol=symbol,
                interval=interval,
                ts=int(ts),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume) if pd.notna(volume) else 0.0,
                adj_close=float(adj) if pd.notna(adj) else None,
            )
        )
    return rows


def _bar_values(df: pd.DataFrame):
    timestamps = pd.DatetimeIndex(df.index).as_unit("s").asi8
    values = df.reindex(
        columns=["Open", "High", "Low", "Close", "Volume", "Adj_close"]
    ).itertuples(index=False, name=None)
    return zip(timestamps, values)


def bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "Adj_close"]
        )
    records = [
        {
            "Date": b.ts,
            "Open": b.open,
            "High": b.high,
            "Low": b.low,
            "Close": b.close,
            "Volume": b.volume,
            "Adj_close": b.adj_close if b.adj_close is not None else b.close,
        }
        for b in bars
    ]
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True)
    return df.set_index("Date").sort_index()


def series_to_float_list(series: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for v in series:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(None)
        else:
            out.append(float(v))
    return out
