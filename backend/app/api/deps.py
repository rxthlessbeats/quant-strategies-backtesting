from typing import Annotated

from fastapi import Depends, Query
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.schemas.requests import AnalysisChartQuery, ChartQuery


def _validated_query(query_type, **values):
    try:
        return query_type(**values)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_context=False)) from exc


def get_chart_query(
    symbol: str = Query(..., description="Ticker symbol"),
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
    interval: str = Query("1d"),
) -> ChartQuery:
    return _validated_query(
        ChartQuery, symbol=symbol, start=start, end=end, interval=interval
    )


def get_analysis_chart_query(
    symbol: str = Query(..., description="Ticker symbol"),
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
    interval: str = Query("1d"),
    indicators: str | None = Query(
        None,
        description="Comma-separated specs, e.g. sma:5,sma:20,ema:50",
    ),
) -> AnalysisChartQuery:
    return _validated_query(
        AnalysisChartQuery,
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        indicators=indicators,
    )


ChartQueryDep = Annotated[ChartQuery, Depends(get_chart_query)]
AnalysisChartQueryDep = Annotated[AnalysisChartQuery, Depends(get_analysis_chart_query)]
