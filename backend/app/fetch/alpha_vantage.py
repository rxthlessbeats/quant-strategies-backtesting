from typing import Any

import pandas as pd
import requests

from app.schemas.settings import settings


class AlphaVantageDownloader:
    """Alpha Vantage time series API with the app's existing dataframe shape."""

    base_url = "https://www.alphavantage.co/query"
    intraday_intervals = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.alpha_vantage_api_key
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required for Alpha Vantage")

    def yahoo(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval == "1d":
            df = self._daily(symbol, outputsize="compact")
        else:
            df = self._intraday(symbol, interval)
        return self._filter_range(df, start_date, end_date)

    def yahoo_max(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        if interval == "1d":
            return self._daily(symbol, outputsize="compact")
        return self._intraday(symbol, interval)

    def search_symbols(self, keywords: str) -> list[dict[str, str | None]]:
        payload = self._request(
            {
                "function": "SYMBOL_SEARCH",
                "keywords": keywords,
            }
        )
        matches = payload.get("bestMatches", [])
        if not isinstance(matches, list):
            return []
        return [
            {
                "symbol": item.get("1. symbol"),
                "name": item.get("2. name"),
                "type": item.get("3. type"),
                "region": item.get("4. region"),
                "currency": item.get("8. currency"),
            }
            for item in matches
            if item.get("1. symbol") and item.get("2. name")
        ]

    def company_overview(self, symbol: str) -> dict[str, Any]:
        payload = self._request(
            {
                "function": "OVERVIEW",
                "symbol": symbol,
            }
        )
        if not payload or "Symbol" not in payload:
            raise ValueError(f"Alpha Vantage overview data missing for {symbol}")
        return payload

    def _daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        payload = self._request(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
            }
        )
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise ValueError(f"Alpha Vantage daily data missing for {symbol}")

        rows = []
        for date, values in series.items():
            rows.append(
                {
                    "Date": pd.Timestamp(date, tz="UTC").normalize(),
                    "Open": self._float(values, "1. open"),
                    "High": self._float(values, "2. high"),
                    "Low": self._float(values, "3. low"),
                    "Close": self._float(values, "4. close"),
                    "Volume": self._float(values, "5. volume"),
                    "Adj_close": self._float(values, "4. close"),
                }
            )
        return pd.DataFrame(rows).set_index("Date").sort_index()

    def _intraday(self, symbol: str, interval: str) -> pd.DataFrame:
        alpha_interval = self.intraday_intervals.get(interval)
        if alpha_interval is None:
            raise ValueError(
                "Alpha Vantage downloader supports intervals: "
                f"{', '.join(sorted(self.intraday_intervals))}"
            )

        payload = self._request(
            {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": alpha_interval,
                "outputsize": "full",
            }
        )
        key = f"Time Series ({alpha_interval})"
        series = payload.get(key)
        if not isinstance(series, dict):
            raise ValueError(f"Alpha Vantage intraday data missing for {symbol}")

        rows = []
        for date, values in series.items():
            rows.append(
                {
                    "Date": pd.Timestamp(date, tz="UTC"),
                    "Open": self._float(values, "1. open"),
                    "High": self._float(values, "2. high"),
                    "Low": self._float(values, "3. low"),
                    "Close": self._float(values, "4. close"),
                    "Volume": self._float(values, "5. volume"),
                }
            )
        return pd.DataFrame(rows).set_index("Date").sort_index()

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        response = requests.get(
            self.base_url,
            params={**params, "apikey": self.api_key},
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(
                "Failed to get Alpha Vantage data: "
                f"Error code {response.status_code}: {response.text}"
            )

        payload = response.json()
        for key in ("Error Message", "Note", "Information"):
            if key in payload:
                raise Exception(f"Alpha Vantage {key}: {payload[key]}")
        return payload

    def _filter_range(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date, tz="UTC")
        end = (
            pd.Timestamp(end_date, tz="UTC")
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )
        return df.loc[(df.index >= start) & (df.index <= end)]

    def _float(self, values: dict[str, str], key: str) -> float:
        return float(values[key])
