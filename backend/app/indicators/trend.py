import pandas as pd

from app.indicators.types import IndicatorEntry, IndicatorMeta


def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Close"].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int = 50) -> pd.Series:
    return df["Close"].ewm(span=period, adjust=False).mean()


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
