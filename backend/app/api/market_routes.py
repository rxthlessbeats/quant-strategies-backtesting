from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.market import (
    CompanyOverviewResponse,
    IndexMetricsResponse,
    PerformanceBenchmarkOptionsResponse,
    PerformanceComparisonResponse,
    TickerSearchResponse,
)
from app.services.company_overview_service import get_company_overview, search_tickers
from app.services.market_metrics_service import get_index_metrics
from app.services.performance_comparison_service import (
    get_performance_comparison,
    list_benchmark_options,
)

router = APIRouter(prefix="/api/v1/market", tags=["market"])


@router.get("/index-metrics", response_model=IndexMetricsResponse)
def get_market_index_metrics(db: Session = Depends(get_db)) -> IndexMetricsResponse:
    try:
        return get_index_metrics(db)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/search", response_model=TickerSearchResponse)
def search_market_tickers(
    keywords: str = Query(
        ..., min_length=1, description="Ticker or company search text"
    ),
) -> TickerSearchResponse:
    try:
        return search_tickers(keywords)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get(
    "/performance-benchmarks",
    response_model=PerformanceBenchmarkOptionsResponse,
)
def get_market_performance_benchmarks() -> PerformanceBenchmarkOptionsResponse:
    return list_benchmark_options()


@router.get(
    "/performance-comparison/{symbol}",
    response_model=PerformanceComparisonResponse,
)
def get_market_performance_comparison(
    symbol: str,
    benchmark: str = Query(
        "SPY",
        description="Benchmark ETF/ticker, e.g. SPY, QQQ, XLK, SMH, or a custom symbol",
    ),
    db: Session = Depends(get_db),
) -> PerformanceComparisonResponse:
    try:
        return get_performance_comparison(db, symbol, benchmark)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/overview/{symbol}", response_model=CompanyOverviewResponse)
def get_market_company_overview(
    symbol: str, db: Session = Depends(get_db)
) -> CompanyOverviewResponse:
    try:
        return get_company_overview(db, symbol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
