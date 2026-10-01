import math
from collections.abc import Callable
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


IndicatorOutput = pd.Series | dict[str, pd.Series]


class IndicatorMeta(BaseModel):
    category: str
    params: dict[str, int | float] = Field(default_factory=dict)
    description: str


class IndicatorEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta: IndicatorMeta
    compute: Callable[..., IndicatorOutput]

    @property
    def category(self) -> str:
        return self.meta.category

    @property
    def params(self) -> dict[str, int | float]:
        return self.meta.params

    @property
    def description(self) -> str:
        return self.meta.description

    def merged_params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        unknown = overrides.keys() - self.meta.params.keys()
        if unknown:
            raise ValueError(
                f"Unsupported indicator parameters: {', '.join(sorted(unknown))}"
            )
        params = {**self.meta.params, **overrides}
        for name, value in params.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 500
            ):
                raise ValueError(f"{name} must be a positive number no larger than 500")
            if name != "std" and int(value) != value:
                raise ValueError(f"{name} must be a whole number of bars")
            if name != "std":
                params[name] = int(value)
        if "fast" in params and params["fast"] >= params["slow"]:
            raise ValueError("MACD fast must be smaller than slow")
        return params
