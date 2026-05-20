from app.indicators.momentum import MOMENTUM
from app.indicators.trend import TREND
from app.indicators.volatility import VOLATILITY
from app.indicators.volume import VOLUME

REGISTRY = {**TREND, **MOMENTUM, **VOLATILITY, **VOLUME}
