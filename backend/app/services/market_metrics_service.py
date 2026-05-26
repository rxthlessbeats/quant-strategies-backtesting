import time

import pandas as pd
from sqlalchemy.orm import Session

from app.schemas.market import IndexMetricItem, IndexMetricsResponse
from app.schemas.requests import ChartQuery
from app.schemas.settings import settings
from app.services.stock_data_service import get_ohlcv


YAHOO_INDEX_CONFIGS = [
    ("spx", "SPX", "^GSPC"),
    ("nasdaq", "NASDAQ", "^IXIC"),
    ("russell2000", "Russell 2000", "^RUT"),
    ("sox", "SOX", "^SOX"),
]

ALPHA_VANTAGE_INDEX_CONFIGS = [
    ("spx", "SPX", "^GSPC", "SPY"),
    ("nasdaq", "NASDAQ", "^IXIC", "QQQ"),
    ("russell2000", "Russell 2000", "^RUT", "IWM"),
    ("sox", "SOX", "^SOX", "SOXX"),
]


def get_index_metrics(db: Session) -> IndexMetricsResponse:
    today = pd.Timestamp.now(tz="UTC")
    end = today.strftime("%Y-%m-%d")
    start = (today - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    configs = _index_configs()
    metrics: list[IndexMetricItem] = []
    for index, (item_id, label, display_symbol, fetch_symbol) in enumerate(configs):
        if index > 0 and settings.data_provider.strip().lower() == "alpha_vantage":
            time.sleep(1.1)
        metric = _index_metric(
            db, item_id, label, display_symbol, fetch_symbol, start, end
        )
        if metric is not None:
            metrics.append(metric)
    return IndexMetricsResponse(metrics=metrics)


def _index_configs() -> list[tuple[str, str, str, str]]:
    if settings.data_provider.strip().lower() == "alpha_vantage":
        return ALPHA_VANTAGE_INDEX_CONFIGS
    return [
        (item_id, label, symbol, symbol)
        for item_id, label, symbol in YAHOO_INDEX_CONFIGS
    ]


def _index_metric(
    db: Session,
    item_id: str,
    label: str,
    display_symbol: str,
    fetch_symbol: str,
    start: str,
    end: str,
) -> IndexMetricItem | None:
    result = get_ohlcv(
        db,
        ChartQuery(symbol=fetch_symbol, start=start, end=end, interval="1d"),
    )
    bars = [bar for bar in result.bars if bar.close is not None]
    if len(bars) < 2:
        return None

    previous = bars[-2]
    latest = bars[-1]
    if previous.close == 0:
        return None

    return IndexMetricItem(
        id=item_id,
        label=label,
        symbol=display_symbol,
        price=latest.close,
        change=(latest.close - previous.close) / previous.close,
        as_of=pd.to_datetime(latest.timestamp, unit="s", utc=True).strftime("%Y-%m-%d"),
    )
