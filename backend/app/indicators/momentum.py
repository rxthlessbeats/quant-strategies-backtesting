import pandas as pd

from app.indicators.types import IndicatorEntry, IndicatorMeta


def momentum(df: pd.DataFrame, period: int = 63) -> pd.Series:
    return df["Close"] / df["Close"].shift(period) - 1.0


MOMENTUM: dict[str, IndicatorEntry] = {
    "momentum": IndicatorEntry(
        meta=IndicatorMeta(
            category="momentum",
            params={"period": 63},
            description="Rate of change: close / close.shift(period) - 1",
        ),
        compute=momentum,
    ),
}
