import time
from datetime import datetime

import pandas as pd
import pandas_datareader.data as web
import requests

from app.schemas.yahoo import YahooChartResponse


class DataDownloader:
    """Yahoo Finance chart API + optional Stooq via pandas-datareader."""

    def __init__(self) -> None:
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            # "User-Agent": random.choice(USER_AGENTS)
        }

    def _timestamp(self, date: str) -> int:
        return int(time.mktime(datetime.strptime(date, "%Y-%m-%d").timetuple()))

    def stooq(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return web.DataReader(symbol, "stooq", start_date, end_date)

    def yahoo(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        start_timestamp = self._timestamp(start_date)
        end_timestamp = self._timestamp(end_date)

        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1={start_timestamp}&period2={end_timestamp}&interval={interval}"
        )
        response = requests.get(url, headers=self.headers, timeout=30)

        if response.status_code == 429:
            raise Exception(f"Rate limit exceeded: {response.text}")
        if response.status_code != 200:
            raise Exception(
                f"Failed to get stock data: Error code {response.status_code}: {response.text}"
            )

        parsed = YahooChartResponse.model_validate(response.json())
        chart = parsed.first_result()
        quote = chart.indicators.quote[0]
        date = pd.to_datetime(chart.timestamp, unit="s")

        if "d" in interval:
            date = date.normalize()
            adj = chart.indicators.adjclose[0] if chart.indicators.adjclose else None
            stock_data = pd.DataFrame(
                {
                    "Date": date,
                    "Open": quote.open,
                    "High": quote.high,
                    "Low": quote.low,
                    "Close": quote.close,
                    "Volume": quote.volume,
                    "Adj_close": adj.adjclose if adj else quote.close,
                }
            )
        elif "h" in interval or "m" in interval:
            stock_data = pd.DataFrame(
                {
                    "Date": date,
                    "Open": quote.open,
                    "High": quote.high,
                    "Low": quote.low,
                    "Close": quote.close,
                    "Volume": quote.volume,
                }
            )
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        stock_data.set_index("Date", inplace=True)
        return stock_data
