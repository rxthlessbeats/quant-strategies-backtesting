from typing import Annotated

from fastapi import Depends, Query

from app.schemas.requests import AnalysisChartQuery, ChartQuery


def get_chart_query(
    symbol: str = Query(..., description="Ticker symbol"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    interval: str = Query("1d"),
) -> ChartQuery:
    return ChartQuery(symbol=symbol, start=start, end=end, interval=interval)


def get_analysis_chart_query(
    symbol: str = Query(..., description="Ticker symbol"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    interval: str = Query("1d"),
    indicators: str | None = Query(
        None,
        description="Comma-separated specs, e.g. sma:5,sma:20,ema:50",
    ),
) -> AnalysisChartQuery:
    return AnalysisChartQuery(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        indicators=indicators,
    )


ChartQueryDep = Annotated[ChartQuery, Depends(get_chart_query)]
AnalysisChartQueryDep = Annotated[AnalysisChartQuery, Depends(get_analysis_chart_query)]
