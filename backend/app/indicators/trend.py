import numpy as np
import pandas as pd

from app.indicators.types import IndicatorEntry, IndicatorMeta


def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Close"].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int = 50) -> pd.Series:
    return df["Close"].ewm(span=period, adjust=False).mean()


def wma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    weights = np.arange(1, period + 1)
    return (
        df["Close"]
        .rolling(period)
        .apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True)
    )


def dema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    first = ema(df, period)
    return 2 * first - first.ewm(span=period, adjust=False).mean()


def tema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    first = ema(df, period)
    second = first.ewm(span=period, adjust=False).mean()
    return 3 * first - 3 * second + second.ewm(span=period, adjust=False).mean()


def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return (df["Close"] * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(
        period
    ).sum().replace(0, np.nan)


TREND: dict[str, IndicatorEntry] = {
    "sma": IndicatorEntry(
        meta=IndicatorMeta(
            category="trend",
            params={"period": 20},
            description="Simple moving average on close",
        ),
        compute=sma,
    ),
    "ema": IndicatorEntry(
        meta=IndicatorMeta(
            category="trend",
            params={"period": 50},
            description="Exponential moving average on close",
        ),
        compute=ema,
    ),
}

for name, compute, description in (
    ("wma", wma, "Weighted moving average, with more weight on recent closes"),
    ("dema", dema, "Double exponential moving average; seeded from the first close"),
    ("tema", tema, "Triple exponential moving average; seeded from the first close"),
    ("vwma", vwma, "Volume-weighted moving average of closing prices"),
):
    TREND[name] = IndicatorEntry(
        meta=IndicatorMeta(
            category="trend", params={"period": 20}, description=description
        ),
        compute=compute,
    )
