import pandas as pd
from sqlalchemy.orm import Session

from app.db import crud
from app.indicators.registry import REGISTRY
from app.schemas.converters import series_to_float_list
from app.schemas.indicators import (
    IndicatorCatalogItem,
    IndicatorSeriesMap,
    IndicatorSpec,
)
from app.schemas.requests import AnalysisChartQuery
from app.schemas.market import OhlcvResult
from app.services.stock_data_service import get_ohlcv


def parse_indicator_specs(spec: str | None) -> list[IndicatorSpec]:
    if not spec or not spec.strip():
        return []
    specs = [IndicatorSpec.parse_spec(part) for part in spec.split(",") if part.strip()]
    for item in specs:
        item.ensure_registered()
    return specs


def compute_indicators(
    df: pd.DataFrame, specs: list[IndicatorSpec]
) -> IndicatorSeriesMap:
    if df.empty or not specs:
        return IndicatorSeriesMap()

    result: dict[str, list[float | None]] = {}
    for spec in specs:
        entry = REGISTRY[spec.name]
        params = entry.merged_params(spec.params.model_dump(exclude_none=True))
        series = entry.compute(df, **params)
        key = _series_key(spec.name, params)
        result[key] = series_to_float_list(series)

    return IndicatorSeriesMap(series=result)


def _series_key(name: str, params: dict) -> str:
    if "period" in params:
        return f"{name}_{params['period']}"
    return name


def list_catalog() -> list[IndicatorCatalogItem]:
    return [
        IndicatorCatalogItem(
            id=name,
            category=entry.category,
            params=entry.params,
            description=entry.description,
        )
        for name, entry in REGISTRY.items()
    ]


def compute_for_query(
    db: Session, query: AnalysisChartQuery
) -> tuple[OhlcvResult, IndicatorSeriesMap]:
    ohlcv = get_ohlcv(db, query)
    specs = parse_indicator_specs(query.indicators)
    start_ts = int(pd.Timestamp(query.start).timestamp())
    end_ts = int(pd.Timestamp(query.end).timestamp())
    df = crud.load_bars_dataframe(db, query.symbol, query.interval, start_ts, end_ts)
    if not df.empty:
        df = df.dropna(subset=["Close"])
    indicators = compute_indicators(df, specs)
    return ohlcv, indicators
