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
