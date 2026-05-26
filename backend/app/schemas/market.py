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
    start: str | None
    end: str | None
    meta: ChartMeta
    bars: list[BarPoint]


class AnalysisChartResponse(BaseModel):
    symbol: str
    interval: str
    start: str | None
    end: str | None
    meta: ChartMeta
    bars: list[BarPoint]
    indicators: dict[str, list[float | None]] = Field(default_factory=dict)


class IndexMetricItem(BaseModel):
    id: str
    label: str
    symbol: str
    price: float
    change: float
    as_of: str


class IndexMetricsResponse(BaseModel):
    metrics: list[IndexMetricItem]


class PerformancePeriodItem(BaseModel):
    id: str
    label: str
    symbol_return: float | None
    benchmark_return: float | None


class PerformanceBenchmarkOption(BaseModel):
    symbol: str
    description: str


class PerformanceBenchmarkGroup(BaseModel):
    category: str
    options: list[PerformanceBenchmarkOption]


class PerformanceBenchmarkOptionsResponse(BaseModel):
    groups: list[PerformanceBenchmarkGroup]


class PerformanceComparisonResponse(BaseModel):
    symbol: str
    benchmark_label: str = "SPY"
    benchmark_symbol: str = "SPY"
    as_of: str
    periods: list[PerformancePeriodItem]


class TickerSearchItem(BaseModel):
    symbol: str
    name: str
    type: str | None = None
    region: str | None = None
    currency: str | None = None


class TickerSearchResponse(BaseModel):
    results: list[TickerSearchItem] = Field(default_factory=list)


class CompanyOverviewResponse(BaseModel):
    symbol: str
    asset_type: str | None = None
    name: str | None = None
    description: str | None = None
    cik: str | None = None
    exchange: str | None = None
    currency: str | None = None
    country: str | None = None
    sector: str | None = None
    industry: str | None = None
    address: str | None = None
    fiscal_year_end: str | None = None
    latest_quarter: str | None = None
    ebitda: str | None = None
    book_value: str | None = None
    dividend_per_share: str | None = None
    eps: str | None = None
    revenue_per_share_ttm: str | None = None
    profit_margin: str | None = None
    operating_margin_ttm: str | None = None
    return_on_assets_ttm: str | None = None
    return_on_equity_ttm: str | None = None
    revenue_ttm: str | None = None
    gross_profit_ttm: str | None = None
    diluted_eps_ttm: str | None = None
    quarterly_earnings_growth_yoy: str | None = None
    quarterly_revenue_growth_yoy: str | None = None
    shares_outstanding: str | None = None
    dividend_date: str | None = None
    ex_dividend_date: str | None = None
    analyst_rating_strong_buy: str | None = None
    analyst_rating_buy: str | None = None
    analyst_rating_hold: str | None = None
    analyst_rating_sell: str | None = None
    analyst_rating_strong_sell: str | None = None
    fetched_at: str | None = None


class OhlcvResult(BaseModel):
    """Internal service result before API serialization."""

    symbol: str
    interval: str
    start: str | None
    end: str | None
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
