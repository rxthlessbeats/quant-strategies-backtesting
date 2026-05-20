from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.market import StockBarsResponse
from app.schemas.requests import ChartQuery
from app.services.stock_data_service import get_ohlcv

router = APIRouter(prefix="/api/v1/stocks", tags=["stocks"])


@router.get("/{symbol}/bars", response_model=StockBarsResponse)
def get_stock_bars(
    symbol: str,
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    interval: str = Query("1d", description="Bar interval: 1d, 1h, etc."),
    db: Session = Depends(get_db),
):
    try:
        query = ChartQuery(symbol=symbol, start=start, end=end, interval=interval)
        result = get_ohlcv(db, query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return result.to_stock_response()
