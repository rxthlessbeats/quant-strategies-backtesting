from pydantic import BaseModel, Field


class BarRow(BaseModel):
    symbol: str = Field(max_length=32)
    interval: str = Field(max_length=8)
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float | None = None

    def to_orm_dict(self) -> dict:
        return self.model_dump()


class FetchMetaRow(BaseModel):
    symbol: str
    interval: str
    last_bar_ts: int | None = None
    fetched_at: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class CompanyFundamentalsRow(BaseModel):
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

    def to_orm_dict(self) -> dict:
        return self.model_dump()
