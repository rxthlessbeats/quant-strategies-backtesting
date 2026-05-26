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
