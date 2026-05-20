from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import validate_date_str


class ChartQuery(BaseModel):
    """Query parameters for OHLCV / analysis chart endpoints."""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=32)
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    interval: str = Field(default="1d", max_length=8)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("start", "end")
    @classmethod
    def validate_dates(cls, value: str) -> str:
        return validate_date_str(value)


class AnalysisChartQuery(ChartQuery):
    indicators: str | None = Field(
        default=None,
        description="Comma-separated specs, e.g. sma:5,sma:20,ema:50",
    )
