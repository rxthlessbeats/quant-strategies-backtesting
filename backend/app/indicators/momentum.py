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
