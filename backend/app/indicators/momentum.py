import numpy as np
import pandas as pd

from app.indicators.types import IndicatorEntry, IndicatorMeta


def momentum(df: pd.DataFrame, period: int = 63) -> pd.Series:
    return df["Close"] / df["Close"].shift(period) - 1.0


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, pd.Series]:
    close = df["Close"]
    macd_line = (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return {
        "line": macd_line,
        "signal": signal_line,
        "hist": hist,
    }


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    return momentum(df, period) * 100


def stoch(
    df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> dict[str, pd.Series]:
    low = df["Low"].rolling(period).min()
    spread = df["High"].rolling(period).max() - low
    raw = ((df["Close"] - low) / spread.replace(0, np.nan) * 100).mask(spread.eq(0), 0)
    k = raw.rolling(smooth_k).mean()
    return {"k": k, "d": k.rolling(smooth_d).mean()}


def willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].rolling(period).max()
    spread = high - df["Low"].rolling(period).min()
    return ((df["Close"] - high) / spread.replace(0, np.nan) * 100).mask(
        spread.eq(0), 0
    )


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    average = typical.rolling(period).mean()
    deviation = typical.rolling(period).apply(
        lambda values: np.abs(values - values.mean()).mean(), raw=True
    )
    return ((typical - average) / (0.015 * deviation.replace(0, np.nan))).mask(
        deviation.eq(0), 0
    )


MOMENTUM: dict[str, IndicatorEntry] = {
    "momentum": IndicatorEntry(
        meta=IndicatorMeta(
            category="momentum",
            params={"period": 63},
            description="Rate of change: close / close.shift(period) - 1",
        ),
        compute=momentum,
    ),
    "rsi": IndicatorEntry(
        meta=IndicatorMeta(
            category="momentum",
            params={"period": 14},
            description="Relative strength index using Wilder-style smoothing",
        ),
        compute=rsi,
    ),
    "macd": IndicatorEntry(
        meta=IndicatorMeta(
            category="momentum",
            params={"fast": 12, "slow": 26, "signal": 9},
            description="Moving average convergence/divergence with signal and histogram",
        ),
        compute=macd,
    ),
}

for name, compute, params, description in (
    ("roc", roc, {"period": 12}, "Rate of change in percent over the lookback window"),
    (
        "stoch",
        stoch,
        {"period": 14, "smooth_k": 3, "smooth_d": 3},
        "Slow stochastic oscillator with smoothed %K and %D lines",
    ),
    (
        "willr",
        willr,
        {"period": 14},
        "Williams %R: close within its high-low range, from -100 to 0",
    ),
    (
        "cci",
        cci,
        {"period": 20},
        "Commodity channel index: typical price versus its mean deviation",
    ),
):
    MOMENTUM[name] = IndicatorEntry(
        meta=IndicatorMeta(category="momentum", params=params, description=description),
        compute=compute,
    )
