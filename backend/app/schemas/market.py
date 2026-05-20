from pydantic import BaseModel, Field

from app.schemas.common import BarPoint, DataSource


class ChartMeta(BaseModel):
    source: DataSource
    cached_through: str | None = None
    fetched_at: str | None = None
    bar_count: int = 0


class StockBarsResponse(BaseModel):
    symbol: str
    interval: str
    start: str
    end: str
    meta: ChartMeta
    bars: list[BarPoint]


class AnalysisChartResponse(BaseModel):
    symbol: str
    interval: str
    start: str
    end: str
    meta: ChartMeta
    bars: list[BarPoint]
    indicators: dict[str, list[float | None]] = Field(default_factory=dict)


class OhlcvResult(BaseModel):
    """Internal service result before API serialization."""

    symbol: str
    interval: str
    start: str
    end: str
    meta: ChartMeta
    bars: list[BarPoint]

    def to_stock_response(self) -> StockBarsResponse:
        return StockBarsResponse(
            symbol=self.symbol,
            interval=self.interval,
            start=self.start,
            end=self.end,
            meta=self.meta,
            bars=self.bars,
        )

    def to_analysis_response(
        self, indicators: dict[str, list[float | None]]
    ) -> AnalysisChartResponse:
        return AnalysisChartResponse(
            symbol=self.symbol,
            interval=self.interval,
            start=self.start,
            end=self.end,
            meta=self.meta,
            bars=self.bars,
            indicators=indicators,
        )
