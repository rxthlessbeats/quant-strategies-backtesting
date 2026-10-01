"""Run: .venv/bin/python scripts/check-indicators.py (no network or live DB writes)."""

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
import pandas as pd

from app.indicators.registry import REGISTRY
from app.schemas.converters import series_to_float_list
from app.services.indicator_service import (
    compute_indicators,
    list_catalog,
    parse_indicator_specs,
)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (
        actual,
        expected,
    )


def check():
    prices = np.array([10.0, 13.0, 11.0, 15.0, 16.0, 14.0])
    df = pd.DataFrame(
        {
            "Open": prices - 0.5,
            "High": prices + 1,
            "Low": prices - 2,
            "Close": prices,
            "Volume": [100.0, 200.0, 150.0, 250.0, 300.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=6, tz="UTC"),
    )
    expected = {
        "sma": 15,
        "wma": 89 / 6,
        "vwma": 9950 / 650,
        "momentum": 3 / 11,
        "roc": 300 / 11,
        "willr": -60,
        "cci": -100,
        "atr": 106 / 27,
        "natr": (106 / 27) / 14 * 100,
        "obv": 600,
        "mfi": 25100 / 292,
        "cmf": 1 / 3,
        "ad": 1100 / 3,
    }
    for name, value in expected.items():
        params = {"period": 3} if "period" in REGISTRY[name].params else {}
        close(REGISTRY[name].compute(df, **params).iloc[-1], value)
    close(REGISTRY["ema"].compute(df, period=2).iloc[1], 12)
    close(REGISTRY["dema"].compute(df, period=2).iloc[1], 38 / 3)
    close(REGISTRY["tema"].compute(df, period=2).iloc[1], 116 / 9)
    for key, value in {"upper": 17, "middle": 14.5, "lower": 12}.items():
        close(REGISTRY["donchian"].compute(df, period=3)[key].iloc[-1], value)
    for key, value in {"upper": 17, "middle": 15, "lower": 13}.items():
        close(REGISTRY["bbands"].compute(df, period=3, std=2)[key].iloc[-1], value)
    stochastic = REGISTRY["stoch"].compute(df, period=3, smooth_k=1, smooth_d=2)
    close(stochastic["k"].iloc[-1], 40)
    close(stochastic["d"].iloc[-1], 63.75)
    assert (
        REGISTRY["atr"].compute(df, period=3).iloc[:3].isna().all()
    ), "ATR seed needs prior closes"

    catalog = list_catalog()
    assert len(catalog) == 21
    assert {item.category for item in catalog} == {
        "trend",
        "momentum",
        "volatility",
        "volume",
    }
    for item in catalog:
        override = {
            key: (2 if key in ("smooth_k", "smooth_d") else 3) for key in item.params
        }
        if item.id == "macd":
            override = {"fast": 2, "slow": 3, "signal": 2}
        query = item.id + (
            ":" + ";".join(f"{key}={value}" for key, value in override.items())
            if override
            else ""
        )
        spec = parse_indicator_specs(query)
        full = compute_indicators(df, spec).series
        prefix = compute_indicators(df.iloc[:4], spec).series
        assert full, item.id
        assert all(len(values) == len(df) for values in full.values()), item.id
        assert {
            key: values[:4] for key, values in full.items()
        } == prefix, f"{item.id} uses future data"
        assert all(
            value is None or math.isfinite(value)
            for values in full.values()
            for value in values
        ), item.id
        # The default API query must also work, including indicators without parameters.
        compute_indicators(df, parse_indicator_specs(item.id))

    flat = df.copy()
    flat[["Open", "High", "Low", "Close"]] = 10.0
    flat["Volume"] = 0.0
    for item in catalog:
        output = REGISTRY[item.id].compute(flat, **item.params)
        series = output.values() if isinstance(output, dict) else [output]
        for values in series:
            assert not np.isinf(values.to_numpy()).any(), item.id
            assert len(series_to_float_list(values)) == len(flat)
    for query in (
        "atr:0",
        "wma:501",
        "obv:20",
        "stoch:smooth_k=2.5",
        "macd:fast=30;slow=12",
        "bbands:std=nan",
        "cci:period=3;unused=1",
        "roc:2.5",
    ):
        try:
            parse_indicator_specs(query)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid parameters accepted: {query}")
    assert not compute_indicators(
        df.iloc[:0], parse_indicator_specs("obv,stoch")
    ).series
    print(
        "PASS: 21 categorized indicators, hand-calculated values, warmup, prefix invariance, flat/zero-volume data and parameter validation"
    )


if __name__ == "__main__":
    check()
