"""Run from the repo root: .venv/bin/python scripts/check-backend-cache.py.

Uses a temporary database and a fake upstream; never changes the live cache.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import sys
import sqlite3
import asyncio
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler

from app.api.deps import get_analysis_chart_query, get_chart_query
from app.db import crud
from app.db.locks import symbol_lock
from app.db.models import Base
from app.fetch import downloader
from app.schemas.common import DataSource
from app.schemas.converters import (
    bar_points_from_dataframe,
    bar_rows_from_dataframe,
    bars_to_dataframe,
)
from app.schemas.db import CompanyFundamentalsRow, MarketDataModuleRow
from app.schemas.requests import AnalysisChartQuery
from app.schemas.settings import settings
from app.services.indicator_service import compute_for_query
from app.services.market_data_service import ensure_modules
from app.services.sync_service import sync_symbol


def frame(start, end, close=100.0):
    dates = pd.date_range(start, end, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 2,
            "Close": close,
            "Volume": 1000.0,
        },
        index=dates,
    )


def check():
    for query in (get_chart_query, get_analysis_chart_query):
        extra = {"indicators": None} if query is get_analysis_chart_query else {}
        normalized = query(
            symbol="nvda", start="2024-1-2", end="2024-1-5", interval="1d", **extra
        )
        assert normalized.start == "2024-01-02" and normalized.end == "2024-01-05"
        for start, end in (
            ("bad", "2024-12-31"),
            ("2024-01-01", None),
            ("2024-12-31", "2024-01-01"),
        ):
            try:
                query(symbol="NVDA", start=start, end=end, interval="1d", **extra)
            except RequestValidationError as exc:
                response = asyncio.run(request_validation_exception_handler(None, exc))
                assert (
                    response.status_code == 422 and json.loads(response.body)["detail"]
                )
                assert all("indicators" not in error["loc"] for error in exc.errors())
            else:
                raise AssertionError("Invalid query parameters must fail validation")
    with TemporaryDirectory() as directory:
        engine = create_engine(
            f"sqlite:///{directory}/check.db", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(engine)
        upstream = Mock()
        upstream.yahoo.side_effect = lambda symbol, start, end, interval: frame(
            start, end
        )
        upstream.yahoo_max.return_value = frame("2024-01-01", "2024-01-10")
        with (
            Session(engine) as db,
            patch("app.services.sync_service.get_downloader", return_value=upstream),
            patch("app.services.sync_service._today", return_value="2024-01-10"),
            patch(
                "app.db.crud.last_expected_daily_ts",
                return_value=int(pd.Timestamp("2024-01-10", tz="UTC").timestamp()),
            ),
        ):
            # Weekend boundaries are covered even though there is no Saturday bar.
            assert (
                sync_symbol(db, "WEEKEND", "2024-01-06", "2024-01-09")
                == DataSource.FETCH
            )
            assert (
                sync_symbol(db, "WEEKEND", "2024-01-06", "2024-01-09")
                == DataSource.CACHE
            )
            assert upstream.yahoo.call_count == 1

            # Disjoint ranges must not pretend the gap was fetched.
            sync_symbol(db, "GAP", "2020-01-01", "2020-01-10")
            sync_symbol(db, "GAP", "2022-01-01", "2022-01-10")
            calls = upstream.yahoo.call_count
            sync_symbol(db, "GAP", "2020-01-01", "2022-01-10")
            assert upstream.yahoo.call_count == calls + 1
            assert upstream.yahoo.call_args.args[1:3] == ("2020-01-01", "2022-01-10")

            # A bounded cache cannot substitute for complete history.
            sync_symbol(db, "GAP")
            assert upstream.yahoo_max.call_count == 1
            assert crud.get_fetch_meta(db, "GAP", "1d").start_date is None
            sync_symbol(db, "GAP")
            assert upstream.yahoo_max.call_count == 1

            # Extending complete history must fetch the gap before a later requested start.
            sync_symbol(db, "EXTEND")
            with (
                patch("app.services.sync_service._today", return_value="2024-01-15"),
                patch("app.db.crud.is_fresh", return_value=False),
            ):
                sync_symbol(db, "EXTEND", "2024-01-15", "2024-01-15")
            assert upstream.yahoo.call_args.args[1:3] == ("2024-01-10", "2024-01-15")

            # Successful empty responses are checked once per refresh interval.
            upstream.yahoo.side_effect = lambda *args: pd.DataFrame()
            sync_symbol(db, "EMPTY", "2024-01-06", "2024-01-07")
            calls = upstream.yahoo.call_count
            sync_symbol(db, "EMPTY", "2024-01-06", "2024-01-07")
            assert upstream.yahoo.call_count == calls

            sync_symbol(db, "HOLIDAY", "2024-01-10", "2024-01-10")
            calls = upstream.yahoo.call_count
            sync_symbol(db, "HOLIDAY", "2024-01-10", "2024-01-10")
            assert upstream.yahoo.call_count == calls
            with patch("app.db.crud.is_fresh", return_value=False):
                sync_symbol(db, "HOLIDAY", "2024-01-10", "2024-01-10")
            assert upstream.yahoo.call_count == calls + 1

            # An upstream failure must not mark stale prices as freshly checked.
            stamp = crud.get_fetch_meta(db, "HOLIDAY", "1d").fetched_at
            upstream.yahoo.side_effect = ValueError("upstream unavailable")
            with patch("app.db.crud.is_fresh", return_value=False):
                try:
                    sync_symbol(db, "HOLIDAY", "2024-01-10", "2024-01-10")
                except ValueError:
                    pass
                else:
                    raise AssertionError("Upstream errors must propagate")
            assert crud.get_fetch_meta(db, "HOLIDAY", "1d").fetched_at == stamp

            # Refresh the last bar in place, and preserve complete-history coverage.
            upstream.yahoo.side_effect = lambda symbol, start, end, interval: frame(
                start, end, 120.0
            )
            meta = crud.get_fetch_meta(db, "GAP", "1d")
            meta.fetched_at = "2024-01-10T14:00:00+00:00"
            db.commit()
            with patch("app.db.crud.is_fresh", return_value=False):
                sync_symbol(db, "GAP")
            assert upstream.yahoo.call_args.args[1] == "2024-01-10"
            assert crud.get_bars(db, "GAP", "1d")[-1].close == 120.0
            assert crud.get_fetch_meta(db, "GAP", "1d").start_date is None

            # Indicator and OHLCV output stay aligned; history is read only once.
            with patch(
                "app.db.crud.load_bars_dataframe", wraps=crud.load_bars_dataframe
            ) as load:
                ohlcv, indicators = compute_for_query(
                    db,
                    AnalysisChartQuery(
                        symbol="GAP",
                        indicators="sma:2,ema:2,momentum:2,rsi:2,macd,bbands:2",
                    ),
                )
            assert load.call_count == 1
            assert indicators.series["sma_2"][-1] == 110.0
            assert all(
                len(values) == len(ohlcv.bars) for values in indicators.series.values()
            )
            with patch("app.services.indicator_service.get_ohlcv_with_frame") as load:
                try:
                    compute_for_query(
                        db, AnalysisChartQuery(symbol="GAP", indicators="unknown")
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unknown indicators must fail")
                load.assert_not_called()

            # No writes or fake newer fetched_at when cached fundamentals are unchanged.
            row = CompanyFundamentalsRow(symbol="GAP", name="Example", eps="1.25")
            saved = crud.upsert_company_fundamentals(db, row)
            stamp = saved.fetched_at
            with patch.object(db, "commit", wraps=db.commit) as commit:
                assert crud.upsert_company_fundamentals(db, row).fetched_at == stamp
                commit.assert_not_called()

            # Fast cached reads bypass a lock held by a fundamentals refresh.
            future_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
            crud.upsert_market_data_module(
                db,
                MarketDataModuleRow(
                    symbol="GAP",
                    module="price",
                    payload_json={"regularMarketPrice": {"raw": 120}},
                    payload_hash="example",
                    next_refresh_at=future_date,
                ),
            )

            def read_cached():
                with Session(engine) as worker_db:
                    assert (
                        ensure_modules(worker_db, "GAP", ["price"])[0]["module"]
                        == "price"
                    )
                    assert sync_symbol(worker_db, "GAP") == DataSource.CACHE

            with ThreadPoolExecutor(max_workers=1) as executor, symbol_lock("GAP"):
                executor.submit(read_cached).result(timeout=2)

            def read_history():
                with Session(engine) as worker_db:
                    return sync_symbol(worker_db, "PARALLEL")

            from time import sleep

            def cold_history(*args):
                sleep(0.05)
                return frame("2024-01-01", "2024-01-10")

            upstream.yahoo_max.side_effect = cold_history
            calls = upstream.yahoo_max.call_count
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: read_history(), range(2)))
            assert upstream.yahoo_max.call_count == calls + 1
            assert sorted(results) == sorted([DataSource.FETCH, DataSource.CACHE])
            with patch(
                "app.services.market_data_service.get_yahoo_downloader"
            ) as yahoo:
                yahoo.return_value.quote_summary.return_value = {
                    "price": {"regularMarketPrice": {"raw": 121}}
                }
                refreshed = ensure_modules(db, "GAP", ["price"], force=True)
                assert refreshed[0]["payload"]["regularMarketPrice"]["raw"] == 121
                yahoo.return_value.quote_summary.assert_called_once()

            # Freshness checks use last successful check, including empty/holiday responses.
            meta = crud.get_fetch_meta(db, "GAP", "1d")
            now = pd.Timestamp.now(tz="UTC")
            meta.last_bar_ts = int(now.normalize().timestamp())
            meta.fetched_at = (now - pd.Timedelta(seconds=10)).isoformat()
            assert crud.is_fresh(db, "GAP", "1d")
            meta.fetched_at = (
                now - pd.Timedelta(seconds=settings.bar_refresh_seconds + 1)
            ).isoformat()
            with patch(
                "app.db.crud._last_expected_daily_date",
                return_value=now.normalize() + pd.Timedelta(days=1),
            ):
                assert not crud.is_fresh(db, "GAP", "1d")
            expected = now.normalize() - pd.Timedelta(days=3)
            close = pd.Timestamp(expected.date(), tz="America/New_York") + pd.Timedelta(
                hours=16
            )
            meta.last_bar_ts = int(expected.timestamp())
            meta.fetched_at = (close - pd.Timedelta(minutes=1)).isoformat()
            with patch("app.db.crud._last_expected_daily_date", return_value=expected):
                assert not crud.is_fresh(db, "GAP", "1d")
                meta.fetched_at = (close + pd.Timedelta(minutes=1)).isoformat()
                assert crud.is_fresh(db, "GAP", "1d")

        # Vectorized timestamp conversion also preserves intraday seconds and null handling.
        df = frame("2024-01-01", "2024-01-03")
        df.index = df.index + pd.Timedelta(hours=14, minutes=30)
        df.iloc[1, df.columns.get_loc("Close")] = float("nan")
        df.iloc[2, df.columns.get_loc("Volume")] = float("nan")
        points = bar_points_from_dataframe(df)
        rows = bar_rows_from_dataframe(df, "ROUNDTRIP", "1h")
        assert len(points) == len(rows) == 2
        assert points[0].timestamp == int(df.index[0].timestamp())
        assert points[-1].volume == 0 and points[-1].adj_close == points[-1].close
        with Session(engine) as db:
            crud.upsert_bars(db, rows)
            assert (
                bar_points_from_dataframe(
                    bars_to_dataframe(crud.get_bars(db, "ROUNDTRIP", "1h"))
                )
                == points
            )
            # Bulk history writes must stay below SQLite's parameter limit.
            db.connection().connection.driver_connection.setlimit(
                sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999
            )
            history = bar_rows_from_dataframe(
                frame("2020-01-01", "2024-01-10"), "LONG", "1d"
            )
            assert crud.upsert_bars(db, history) == len(history)
            history[-1].close = 130.0
            crud.upsert_bars(db, history)
            assert len(crud.get_bars(db, "LONG", "1d")) == len(history)
            assert crud.get_bars(db, "LONG", "1d")[-1].close == 130.0

        # Sessions/crumbs are reused within a worker, never shared across workers.
        with patch.object(settings, "data_provider", "yahoo"):
            first = downloader.get_downloader()
            assert downloader.get_downloader() is first
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(downloader.get_downloader).result() is not first
        with (
            patch.object(settings, "data_provider", "alpha_vantage"),
            patch.object(settings, "alpha_vantage_api_key", "test"),
        ):
            assert isinstance(
                downloader.get_downloader(), downloader.AlphaVantageDownloader
            )
            assert downloader.get_yahoo_downloader() is first
        engine.dispose()
    print(
        "PASS: cache coverage, freshness, indicators, forced refresh, independent locks, providers and timestamp roundtrip"
    )


if __name__ == "__main__":
    check()
