from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analysis_routes, market_routes, stock_routes
from app.db.database import init_db
from app.schemas.health import HealthResponse
from app.services.market_data_scheduler import (
    start_market_data_scheduler,
    stop_market_data_scheduler,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_market_data_scheduler()
    try:
        yield
    finally:
        stop_market_data_scheduler()


app = FastAPI(
    title="RookieTrader API",
    description="Stock OHLCV cache and technical indicators",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(stock_routes.router)
app.include_router(analysis_routes.router)
app.include_router(market_routes.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()
