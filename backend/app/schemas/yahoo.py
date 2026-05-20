"""Pydantic models for Yahoo Finance chart API JSON."""

from pydantic import BaseModel, Field


class YahooQuote(BaseModel):
    open: list[float | None] = Field(default_factory=list)
    high: list[float | None] = Field(default_factory=list)
    low: list[float | None] = Field(default_factory=list)
    close: list[float | None] = Field(default_factory=list)
    volume: list[float | None] = Field(default_factory=list)


class YahooAdjClose(BaseModel):
    adjclose: list[float | None] = Field(default_factory=list)


class YahooIndicators(BaseModel):
    quote: list[YahooQuote]
    adjclose: list[YahooAdjClose] | None = None


class YahooChartResult(BaseModel):
    timestamp: list[int]
    indicators: YahooIndicators


class YahooChartResponse(BaseModel):
    chart: dict

    def first_result(self) -> YahooChartResult:
        result = self.chart["result"][0]
        return YahooChartResult.model_validate(result)
