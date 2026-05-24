from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import validate_date_str


class ChartQuery(BaseModel):
    """Query parameters for OHLCV / analysis chart endpoints."""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=32)
    start: str | None = Field(default=None, description="YYYY-MM-DD")
    end: str | None = Field(default=None, description="YYYY-MM-DD")
    interval: str = Field(default="1d", max_length=8)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("start", "end")
    @classmethod
    def validate_dates(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_date_str(value)

    @model_validator(mode="after")
    def validate_date_pair(self) -> "ChartQuery":
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together")
        return self


class AnalysisChartQuery(ChartQuery):
    indicators: str | None = Field(
        default=None,
        description="Comma-separated specs, e.g. sma:5,sma:20,ema:50",
    )
