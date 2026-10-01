import numpy as np
import pandas as pd

from app.indicators.types import IndicatorEntry, IndicatorMeta


def obv(df: pd.DataFrame) -> pd.Series:
    change = df["Close"].diff()
    direction = np.sign(change).fillna(1)
    return (direction * df["Volume"]).cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    change = typical.diff()
    flow = typical * df["Volume"]
    positive = flow.where(change.gt(0), 0).mask(change.isna())
    negative = flow.where(change.lt(0), 0).mask(change.isna())
    positive = positive.rolling(period).sum()
    total = positive + negative.rolling(period).sum()
    return (100 * positive / total.replace(0, np.nan)).mask(total.eq(0), 0)


def _money_flow_volume(df: pd.DataFrame) -> pd.Series:
    spread = df["High"] - df["Low"]
    multiplier = (
        (2 * df["Close"] - df["High"] - df["Low"]) / spread.replace(0, np.nan)
    ).mask(spread.eq(0), 0)
    return multiplier * df["Volume"]


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["Volume"].rolling(period).sum()
    return (
        _money_flow_volume(df).rolling(period).sum() / volume.replace(0, np.nan)
    ).mask(volume.eq(0), 0)


def ad(df: pd.DataFrame) -> pd.Series:
    return _money_flow_volume(df).cumsum()


VOLUME: dict[str, IndicatorEntry] = {}
for name, compute, params, description in (
    (
        "obv",
        obv,
        {},
        "On-balance volume: cumulative signed volume, starting at first-bar volume",
    ),
    (
        "mfi",
        mfi,
        {"period": 14},
        "Money flow index: volume-weighted momentum from 0 to 100",
    ),
    (
        "cmf",
        cmf,
        {"period": 20},
        "Chaikin money flow: rolling volume-weighted accumulation and distribution",
    ),
    ("ad", ad, {}, "Accumulation/distribution line: cumulative money flow volume"),
):
    VOLUME[name] = IndicatorEntry(
        meta=IndicatorMeta(category="volume", params=params, description=description),
        compute=compute,
    )
