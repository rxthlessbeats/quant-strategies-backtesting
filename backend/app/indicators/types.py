from collections.abc import Callable
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class IndicatorMeta(BaseModel):
    category: str
    params: dict[str, int | float] = Field(default_factory=dict)
    description: str


class IndicatorEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta: IndicatorMeta
    compute: Callable[..., pd.Series]

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
        return {**self.meta.params, **overrides}
