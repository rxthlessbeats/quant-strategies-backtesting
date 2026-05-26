from typing import Any

from pydantic import BaseModel, Field, field_validator


class IndicatorParams(BaseModel):
    period: int | None = None

    model_config = {"extra": "allow"}


class IndicatorSpec(BaseModel):
    name: str
    params: IndicatorParams = Field(default_factory=IndicatorParams)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return value.strip().lower()

    @classmethod
    def parse_spec(cls, raw: str) -> "IndicatorSpec":
        raw = raw.strip()
        if ":" in raw:
            name, param_str = raw.split(":", 1)
            params: dict[str, Any] = {}
            if param_str.isdigit():
                params["period"] = int(param_str)
            else:
                for kv in param_str.split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        params[k.strip()] = _parse_param_value(v)
            return cls(name=name, params=IndicatorParams(**params))
        return cls(name=raw)

    def ensure_registered(self) -> None:
        from app.indicators.registry import REGISTRY

        if self.name not in REGISTRY:
            raise ValueError(f"Unknown indicator: {self.name}")


def _parse_param_value(value: str) -> int | float | str:
    stripped = value.strip()
    try:
        numeric = float(stripped)
    except ValueError:
        return stripped
    if numeric.is_integer():
        return int(numeric)
    return numeric


class IndicatorCatalogItem(BaseModel):
    id: str
    category: str
    params: dict[str, int | float]
    description: str


class IndicatorSeriesMap(BaseModel):
    series: dict[str, list[float | None]] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> list[float | None]:
        return self.series[key]

    def as_dict(self) -> dict[str, list[float | None]]:
        return self.series
