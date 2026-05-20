import math

import pandas as pd

from app.db.models import Bar
from app.schemas.common import BarPoint
from app.schemas.db import BarRow


def bar_points_from_dataframe(df: pd.DataFrame) -> list[BarPoint]:
    bars: list[BarPoint] = []
    for ts, row in df.iterrows():
        if pd.isna(row.get("Close")):
            continue
        adj = row.get("Adj_close")
        bars.append(
            BarPoint(
                timestamp=int(pd.Timestamp(ts).timestamp()),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume") or 0),
                adj_close=float(adj) if pd.notna(adj) else float(row["Close"]),
            )
        )
    return bars


def bar_rows_from_dataframe(
    df: pd.DataFrame, symbol: str, interval: str
) -> list[BarRow]:
    rows: list[BarRow] = []
    for ts, row in df.iterrows():
        if pd.isna(row.get("Close")):
            continue
        adj = row.get("Adj_close")
        rows.append(
            BarRow(
                symbol=symbol,
                interval=interval,
                ts=int(pd.Timestamp(ts).timestamp()),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume") or 0),
                adj_close=float(adj) if pd.notna(adj) else None,
            )
        )
    return rows


def bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "Adj_close"]
        )
    records = [
        {
            "Date": pd.to_datetime(b.ts, unit="s", utc=True),
            "Open": b.open,
            "High": b.high,
            "Low": b.low,
            "Close": b.close,
            "Volume": b.volume,
            "Adj_close": b.adj_close if b.adj_close is not None else b.close,
        }
        for b in bars
    ]
    return pd.DataFrame(records).set_index("Date").sort_index()


def series_to_float_list(series: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for v in series:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(None)
        else:
            out.append(float(v))
    return out
