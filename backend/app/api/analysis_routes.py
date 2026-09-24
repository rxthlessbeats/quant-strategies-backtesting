from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import AnalysisChartQueryDep
from app.api.http_errors import http_502
from app.db.database import get_db
from app.schemas.indicators import IndicatorCatalogItem
from app.schemas.market import AnalysisChartResponse
from app.services.indicator_service import compute_for_query, list_catalog

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


@router.get("/chart", response_model=AnalysisChartResponse)
def get_analysis_chart(
    query: AnalysisChartQueryDep,
    db: Session = Depends(get_db),
):
    try:
        ohlcv, indicators = compute_for_query(db, query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise http_502(e, fallback="Failed to load chart data") from e

    return ohlcv.to_analysis_response(indicators.as_dict())


@router.get("/indicators", response_model=list[IndicatorCatalogItem])
def get_indicator_catalog() -> list[IndicatorCatalogItem]:
    return list_catalog()
