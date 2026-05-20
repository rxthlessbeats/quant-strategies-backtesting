from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    CACHE = "cache"
    FETCH = "fetch"


class BarPoint(BaseModel):
    timestamp: int = Field(description="Unix timestamp (seconds)")
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float | None = None


def validate_date_str(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d")
    return value
