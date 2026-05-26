import pandas as pd
from sqlalchemy.orm import Session

from app.schemas.common import BarPoint
from app.schemas.market import (
    PerformanceBenchmarkGroup,
    PerformanceBenchmarkOption,
    PerformanceBenchmarkOptionsResponse,
    PerformanceComparisonResponse,
    PerformancePeriodItem,
)
from app.schemas.requests import ChartQuery
from app.services.stock_data_service import get_ohlcv

BENCHMARK_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Broad Market": [
        ("SPY", "S&P 500"),
        ("QQQ", "Nasdaq 100 / Growth"),
    ],
    "Technology": [
        ("XLK", "Technology"),
        ("SMH", "Semiconductors"),
        ("SOXX", "Semiconductors"),
        ("IGV", "Software"),
    ],
    "Consumer & Communication": [
        ("XLC", "Communication Services"),
        ("XLY", "Consumer Discretionary"),
        ("XLP", "Consumer Staples"),
    ],
    "Financials": [
        ("XLF", "Financials"),
        ("KBE", "Banks"),
        ("KRE", "Regional Banks"),
    ],
    "Healthcare": [
        ("XLV", "Healthcare"),
        ("IBB", "Biotech"),
        ("XBI", "Biotech"),
    ],
    "Industrials": [
        ("XLI", "Industrials"),
        ("ITA", "Aerospace & Defense"),
        ("IYT", "Transportation"),
    ],
    "Energy, Materials & Utilities": [
        ("XLE", "Energy"),
        ("XLB", "Materials"),
        ("XME", "Metals & Mining"),
        ("XLU", "Utilities"),
    ],
    "Real Estate & Housing": [
        ("XLRE", "Real Estate"),
        ("VNQ", "REITs"),
        ("ITB", "Homebuilders"),
        ("XHB", "Homebuilders"),
    ],
}

PERIODS = [
    ("1w", "1W", pd.DateOffset(days=7)),
    ("1m", "1M", pd.DateOffset(months=1)),
    ("1q", "1Q", pd.DateOffset(months=3)),
    ("6m", "6M", pd.DateOffset(months=6)),
    ("ytd", "YTD", None),
    ("1y", "1Y", pd.DateOffset(years=1)),
    ("3y", "3Y", pd.DateOffset(years=3)),
    ("5y", "5Y", pd.DateOffset(years=5)),
]


def list_benchmark_options() -> PerformanceBenchmarkOptionsResponse:
    return PerformanceBenchmarkOptionsResponse(
        groups=[
            PerformanceBenchmarkGroup(
                category=category,
                options=[
                    PerformanceBenchmarkOption(symbol=symbol, description=description)
                    for symbol, description in options
                ],
            )
            for category, options in BENCHMARK_GROUPS.items()
        ]
    )


def get_performance_comparison(
    db: Session, symbol: str, benchmark: str = "SPY"
) -> PerformanceComparisonResponse:
    benchmark_key = benchmark.strip().upper()
    if not benchmark_key:
        raise ValueError(f"Invalid benchmark symbol '{benchmark}'")
    benchmark_label = benchmark_key
    benchmark_symbol = benchmark_key
    today = pd.Timestamp.now(tz="UTC").normalize()
    start = (today - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    symbol = symbol.strip().upper()
    symbol_series = _close_series(
        get_ohlcv(
            db, ChartQuery(symbol=symbol, start=start, end=end, interval="1d")
        ).bars
    )
    benchmark_series = _close_series(
        get_ohlcv(
            db,
            ChartQuery(
                symbol=benchmark_symbol,
                start=start,
                end=end,
                interval="1d",
            ),
        ).bars
    )

    as_of = _comparison_as_of(symbol_series, benchmark_series)
    periods = [
        PerformancePeriodItem(
            id=period_id,
            label=label,
            symbol_return=_period_return(symbol_series, as_of, offset),
            benchmark_return=_period_return(benchmark_series, as_of, offset),
        )
        for period_id, label, offset in PERIODS
    ]

    return PerformanceComparisonResponse(
        symbol=symbol,
        benchmark_label=benchmark_label,
        benchmark_symbol=benchmark_symbol,
        as_of=as_of.strftime("%Y-%m-%d") if as_of is not None else "",
        periods=periods,
    )


def _close_series(bars: list[BarPoint]) -> pd.Series:
    if not bars:
        return pd.Series(dtype="float64")

    series = pd.Series(
        data=[bar.close for bar in bars],
        index=[
            pd.to_datetime(bar.timestamp, unit="s", utc=True).normalize()
            for bar in bars
        ],
        dtype="float64",
    )
    return series.sort_index().groupby(level=0).last()


def _comparison_as_of(
    symbol_series: pd.Series, benchmark_series: pd.Series
) -> pd.Timestamp | None:
    if symbol_series.empty or benchmark_series.empty:
        return None

    return min(symbol_series.index[-1], benchmark_series.index[-1])


def _period_return(
    series: pd.Series, as_of: pd.Timestamp | None, offset: pd.DateOffset | None
) -> float | None:
    if series.empty or as_of is None:
        return None

    latest = _close_on_or_before(series, as_of)
    target = _period_target(as_of, offset)
    reference = _close_on_or_before(series, target)
    if latest is None or reference is None or reference == 0:
        return None

    return latest / reference - 1


def _period_target(as_of: pd.Timestamp, offset: pd.DateOffset | None) -> pd.Timestamp:
    if offset is None:
        return pd.Timestamp(year=as_of.year - 1, month=12, day=31, tz="UTC")
    return as_of - offset


def _close_on_or_before(series: pd.Series, date: pd.Timestamp) -> float | None:
    values = series.loc[:date]
    if values.empty:
        return None
    return float(values.iloc[-1])
