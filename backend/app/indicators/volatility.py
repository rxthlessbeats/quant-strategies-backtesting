import numpy as np
import pandas as pd

from app.indicators.types import IndicatorEntry, IndicatorMeta


def bbands(
    df: pd.DataFrame, period: int = 20, std: int | float = 2
) -> dict[str, pd.Series]:
    middle = df["Close"].rolling(window=period).mean()
    deviation = df["Close"].rolling(window=period).std()
    upper = middle + deviation * std
    lower = middle - deviation * std
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = df["Close"].shift()
    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous).abs(),
            (df["Low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    seed = ranges.iloc[1 : period + 1].mean()
    ranges.iloc[:period] = np.nan
    if len(ranges) > period:
        ranges.iloc[period] = seed
    return ranges.ewm(alpha=1 / period, adjust=False).mean()


def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return atr(df, period) / df["Close"].replace(0, np.nan) * 100


def donchian(df: pd.DataFrame, period: int = 20) -> dict[str, pd.Series]:
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    return {"upper": upper, "middle": (upper + lower) / 2, "lower": lower}


VOLATILITY: dict[str, IndicatorEntry] = {
    "bbands": IndicatorEntry(
        meta=IndicatorMeta(
            category="volatility",
            params={"period": 20, "std": 2},
            description="Bollinger Bands using close rolling mean and standard deviation",
        ),
        compute=bbands,
    ),
}

for name, compute, period, description in (
    (
        "atr",
        atr,
        14,
        "Average true range, seeded with a simple average then Wilder-smoothed",
    ),
    ("natr", natr, 14, "Normalized average true range as a percentage of close"),
    (
        "donchian",
        donchian,
        20,
        "Donchian channels: highest high, lowest low, and their midpoint",
    ),
):
    VOLATILITY[name] = IndicatorEntry(
        meta=IndicatorMeta(
            category="volatility", params={"period": period}, description=description
        ),
        compute=compute,
    )
