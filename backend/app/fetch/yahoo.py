import time
from datetime import datetime
from typing import Any

import pandas as pd
import pandas_datareader.data as web
import requests

from app.schemas.yahoo import YahooChartResponse

US_EXCHANGE_CODES = frozenset({"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "PNK"})
US_EXCHANGE_LABELS = frozenset(
    {"NASDAQ", "NYSE", "NYSE ARCA", "AMEX", "NYSEAMERICAN", "BATS"}
)


def _is_us_equity_quote(item: dict) -> bool:
    if item.get("quoteType") != "EQUITY":
        return False
    symbol = item.get("symbol") or ""
    if "." in symbol:
        return False
    exchange = (item.get("exchDisp") or item.get("exchange") or "").upper()
    if exchange in US_EXCHANGE_CODES:
        return True
    return any(label in exchange for label in US_EXCHANGE_LABELS)


class DataDownloader:
    """Yahoo Finance chart API + optional Stooq via pandas-datareader."""

    def __init__(self) -> None:
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            # "User-Agent": random.choice(USER_AGENTS)
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._crumb: str | None = None

    def _timestamp(self, date: str) -> int:
        return int(time.mktime(datetime.strptime(date, "%Y-%m-%d").timetuple()))

    def search_symbols(self, keywords: str) -> list[dict[str, str | None]]:
        response = self.session.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": keywords, "quotes_count": 15, "news_count": 0},
            timeout=30,
        )
        self._raise_for_status(response, "Yahoo search")
        quotes = response.json().get("quotes", [])
        if not isinstance(quotes, list):
            return []
        results: list[dict[str, str | None]] = []
        for item in quotes:
            if not _is_us_equity_quote(item):
                continue
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname")
            if not symbol or not name:
                continue
            results.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "type": item.get("quoteType"),
                    "region": item.get("exchDisp") or item.get("exchange"),
                    "currency": None,
                }
            )
            if len(results) >= 8:
                break
        return results

    def company_overview(self, symbol: str) -> dict[str, Any]:
        modules = ",".join(
            [
                "assetProfile",
                "summaryProfile",
                "defaultKeyStatistics",
                "financialData",
                "summaryDetail",
                "calendarEvents",
                "price",
            ]
        )
        response = self.session.get(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
            params={"modules": modules, "crumb": self._get_crumb()},
            timeout=30,
        )
        self._raise_for_status(response, "Yahoo quote summary")
        result = response.json().get("quoteSummary", {}).get("result")
        if not result:
            raise ValueError(f"Yahoo overview data missing for {symbol}")
        return self._overview_from_quote_summary(symbol, result[0])

    def _raise_for_status(self, response: requests.Response, source: str) -> None:
        if response.status_code == 429:
            raise Exception(f"{source} rate limit exceeded: {response.text}")
        if response.status_code != 200:
            raise Exception(
                f"{source} failed: Error code {response.status_code}: {response.text}"
            )

    def _get_crumb(self) -> str:
        if self._crumb:
            return self._crumb
        self.session.get("https://fc.yahoo.com", timeout=30)
        response = self.session.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            timeout=30,
        )
        self._raise_for_status(response, "Yahoo crumb")
        self._crumb = response.text.strip()
        if not self._crumb:
            raise ValueError("Yahoo crumb response was empty")
        return self._crumb

    def _overview_from_quote_summary(
        self, symbol: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        profile = data.get("assetProfile") or data.get("summaryProfile") or {}
        stats = data.get("defaultKeyStatistics") or {}
        financial = data.get("financialData") or {}
        detail = data.get("summaryDetail") or {}
        calendar = data.get("calendarEvents") or {}
        price = data.get("price") or {}

        address = ", ".join(
            part
            for part in [
                profile.get("address1"),
                profile.get("city"),
                profile.get("state"),
                profile.get("zip"),
            ]
            if part
        )

        return {
            "Symbol": symbol.upper(),
            "AssetType": price.get("quoteType") or price.get("typeDisp"),
            "Name": self._raw(price.get("longName"))
            or self._raw(price.get("shortName"))
            or symbol.upper(),
            "Description": profile.get("longBusinessSummary"),
            "Exchange": self._raw(price.get("exchangeName"))
            or self._raw(price.get("exchange")),
            "Currency": self._raw(price.get("currency")),
            "Country": profile.get("country"),
            "Sector": profile.get("sector"),
            "Industry": profile.get("industry"),
            "Address": address or None,
            "FiscalYearEnd": self._raw(calendar.get("fiscalYearEnd")),
            "LatestQuarter": self._raw(calendar.get("earningsDate")),
            "EBITDA": self._raw(financial.get("ebitda")),
            "BookValue": self._raw(stats.get("bookValue")),
            "DividendPerShare": self._raw(detail.get("dividendRate")),
            "EPS": self._raw(stats.get("trailingEps")),
            "RevenuePerShareTTM": self._raw(financial.get("revenuePerShare")),
            "ProfitMargin": self._raw(financial.get("profitMargins")),
            "OperatingMarginTTM": self._raw(financial.get("operatingMargins")),
            "ReturnOnAssetsTTM": self._raw(financial.get("returnOnAssets")),
            "ReturnOnEquityTTM": self._raw(financial.get("returnOnEquity")),
            "RevenueTTM": self._raw(financial.get("totalRevenue")),
            "GrossProfitTTM": self._raw(financial.get("grossProfits")),
            "DilutedEPSTTM": self._raw(stats.get("trailingEps")),
            "QuarterlyEarningsGrowthYOY": self._raw(financial.get("earningsGrowth")),
            "QuarterlyRevenueGrowthYOY": self._raw(financial.get("revenueGrowth")),
            "SharesOutstanding": self._raw(stats.get("sharesOutstanding")),
            "DividendDate": self._raw(calendar.get("dividendDate")),
            "ExDividendDate": self._raw(calendar.get("exDividendDate")),
        }

    def _raw(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, dict):
            value = value.get("raw") if "raw" in value else value.get("fmt")
        if value is None:
            return None
        return str(value)

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
        response = self.session.get(url, timeout=30)

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

    def yahoo_max(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        end_timestamp = int(time.time())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1=0&period2={end_timestamp}&interval={interval}"
        )
        response = self.session.get(url, timeout=30)

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
