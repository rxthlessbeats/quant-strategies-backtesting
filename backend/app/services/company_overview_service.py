from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.db import crud
from app.fetch.yahoo import DataDownloader
from app.schemas.db import CompanyFundamentalsRow
from app.schemas.market import (
    CompanyOverviewResponse,
    TickerSearchItem,
    TickerSearchResponse,
)

FUNDAMENTALS_TTL = timedelta(hours=24)

OVERVIEW_FIELD_MAP = {
    "Symbol": "symbol",
    "AssetType": "asset_type",
    "Name": "name",
    "Description": "description",
    "CIK": "cik",
    "Exchange": "exchange",
    "Currency": "currency",
    "Country": "country",
    "Sector": "sector",
    "Industry": "industry",
    "Address": "address",
    "FiscalYearEnd": "fiscal_year_end",
    "LatestQuarter": "latest_quarter",
    "EBITDA": "ebitda",
    "BookValue": "book_value",
    "DividendPerShare": "dividend_per_share",
    "EPS": "eps",
    "RevenuePerShareTTM": "revenue_per_share_ttm",
    "ProfitMargin": "profit_margin",
    "OperatingMarginTTM": "operating_margin_ttm",
    "ReturnOnAssetsTTM": "return_on_assets_ttm",
    "ReturnOnEquityTTM": "return_on_equity_ttm",
    "RevenueTTM": "revenue_ttm",
    "GrossProfitTTM": "gross_profit_ttm",
    "DilutedEPSTTM": "diluted_eps_ttm",
    "QuarterlyEarningsGrowthYOY": "quarterly_earnings_growth_yoy",
    "QuarterlyRevenueGrowthYOY": "quarterly_revenue_growth_yoy",
    "SharesOutstanding": "shares_outstanding",
    "DividendDate": "dividend_date",
    "ExDividendDate": "ex_dividend_date",
    "AnalystRatingStrongBuy": "analyst_rating_strong_buy",
    "AnalystRatingBuy": "analyst_rating_buy",
    "AnalystRatingHold": "analyst_rating_hold",
    "AnalystRatingSell": "analyst_rating_sell",
    "AnalystRatingStrongSell": "analyst_rating_strong_sell",
}


def search_tickers(keywords: str) -> TickerSearchResponse:
    downloader = DataDownloader()
    results = [
        TickerSearchItem(**item)
        for item in downloader.search_symbols(keywords)
        if item.get("symbol") and item.get("name")
    ]
    return TickerSearchResponse(results=results)


def get_company_overview(db: Session, symbol: str) -> CompanyOverviewResponse:
    normalized = symbol.upper()
    cached = crud.get_company_fundamentals(db, normalized)
    if cached is not None and _is_recent(cached.fetched_at):
        return _response_from_row(crud.company_fundamentals_to_schema(cached))

    downloader = DataDownloader()
    payload = downloader.company_overview(normalized)
    row = _row_from_overview(payload, normalized)
    saved = crud.upsert_company_fundamentals(db, row)
    return _response_from_row(crud.company_fundamentals_to_schema(saved))


def _is_recent(fetched_at: str | None) -> bool:
    if not fetched_at:
        return False
    try:
        ts = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < FUNDAMENTALS_TTL


def _row_from_overview(payload: dict, fallback_symbol: str) -> CompanyFundamentalsRow:
    values = {
        target: _clean_value(payload.get(source))
        for source, target in OVERVIEW_FIELD_MAP.items()
    }
    values["symbol"] = values.get("symbol") or fallback_symbol
    return CompanyFundamentalsRow(**values)


def _response_from_row(row: CompanyFundamentalsRow | None) -> CompanyOverviewResponse:
    if row is None:
        raise ValueError("Company overview not found")
    return CompanyOverviewResponse(**row.model_dump())


def _clean_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "-"}:
        return None
    return text
