from sqlalchemy import BigInteger, Float, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Bar(Base):
    __tablename__ = "bars"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "ts", name="uq_bar_symbol_interval_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[int] = mapped_column(BigInteger, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)


class FetchMeta(Base):
    __tablename__ = "fetch_meta"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(8), primary_key=True)
    last_bar_ts: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    fetched_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(16), nullable=True)


class MarketDataModule(Base):
    __tablename__ = "market_data_modules"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "module", name="uq_market_data_module_symbol_module"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    module: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[dict] = mapped_column(JSON)
    payload_hash: Mapped[str] = mapped_column(String(64), index=True)
    last_checked_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_changed_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fetched_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    next_refresh_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latest_event_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="yahoo")
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)


class CompanyFundamentals(Base):
    __tablename__ = "company_fundamentals"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    asset_type: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    cik: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    address: Mapped[str | None] = mapped_column(String, nullable=True)
    fiscal_year_end: Mapped[str | None] = mapped_column(String, nullable=True)
    latest_quarter: Mapped[str | None] = mapped_column(String, nullable=True)
    ebitda: Mapped[str | None] = mapped_column(String, nullable=True)
    book_value: Mapped[str | None] = mapped_column(String, nullable=True)
    dividend_per_share: Mapped[str | None] = mapped_column(String, nullable=True)
    eps: Mapped[str | None] = mapped_column(String, nullable=True)
    revenue_per_share_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    profit_margin: Mapped[str | None] = mapped_column(String, nullable=True)
    operating_margin_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    return_on_assets_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    return_on_equity_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    revenue_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    gross_profit_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    diluted_eps_ttm: Mapped[str | None] = mapped_column(String, nullable=True)
    quarterly_earnings_growth_yoy: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    quarterly_revenue_growth_yoy: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    shares_outstanding: Mapped[str | None] = mapped_column(String, nullable=True)
    dividend_date: Mapped[str | None] = mapped_column(String, nullable=True)
    ex_dividend_date: Mapped[str | None] = mapped_column(String, nullable=True)
    analyst_rating_strong_buy: Mapped[str | None] = mapped_column(String, nullable=True)
    analyst_rating_buy: Mapped[str | None] = mapped_column(String, nullable=True)
    analyst_rating_hold: Mapped[str | None] = mapped_column(String, nullable=True)
    analyst_rating_sell: Mapped[str | None] = mapped_column(String, nullable=True)
    analyst_rating_strong_sell: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    fetched_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
