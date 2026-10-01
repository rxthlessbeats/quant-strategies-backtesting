import pandas as pd
from sqlalchemy.orm import Session

from app.indicators.registry import REGISTRY
from app.schemas.converters import series_to_float_list
from app.schemas.indicators import (
    IndicatorCatalogItem,
    IndicatorSeriesMap,
    IndicatorSpec,
)
from app.schemas.requests import AnalysisChartQuery
from app.schemas.market import OhlcvResult
from app.services.stock_data_service import get_ohlcv_with_frame


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
        output = entry.compute(df, **params)
        if isinstance(output, pd.Series):
            result[_series_key(spec.name, params)] = series_to_float_list(output)
        else:
            for part, series in output.items():
                result[_series_key(spec.name, params, part)] = series_to_float_list(
                    series
                )

    return IndicatorSeriesMap(series=result)


def _series_key(name: str, params: dict, part: str | None = None) -> str:
    suffix = _params_suffix(name, params)
    if part is not None:
        if name == "macd" and part == "line":
            return f"{name}_{suffix}"
        return f"{name}_{part}_{suffix}"
    if suffix:
        return f"{name}_{suffix}"
    return name


def _params_suffix(name: str, params: dict) -> str:
    return "_".join(str(value) for value in params.values())


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
    specs = parse_indicator_specs(query.indicators)
    ohlcv, df = get_ohlcv_with_frame(db, query)
    indicators = compute_indicators(df, specs)
    return ohlcv, indicators
