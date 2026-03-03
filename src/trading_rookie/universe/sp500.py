"""
S&P 500–style universe construction: sector map, momentum scoring, and sector-balanced selection.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from trading_rookie.config.const import SECTOR_MAP


class SectorMap:
    """Maps tickers to GICS sectors from config. Builds ticker -> sector and exposes full ticker list."""

    def __init__(self, sector_map: Optional[dict[str, list[str]]] = None) -> None:
        self._sector_map = sector_map if sector_map is not None else SECTOR_MAP
        self._ticker_to_sector = self._build()

    def _build(self) -> dict[str, str]:
        out = {}
        for sector, tickers in self._sector_map.items():
            for t in tickers:
                out[t] = sector
        return out

    def get_sector(self, ticker: str) -> Optional[str]:
        return self._ticker_to_sector.get(ticker)

    def all_tickers(self) -> list[str]:
        return [t for tickers in self._sector_map.values() for t in tickers][:11]

    @property
    def ticker_to_sector(self) -> dict[str, str]:
        return self._ticker_to_sector.copy()


class MomentumScorer:
    """Scores tickers by momentum compatibility: Sharpe of a simple momentum strategy minus penalty for max drawdown."""

    def __init__(
        self, lookback: int = 63, annualize: int = 252, dd_penalty: float = 0.5
    ) -> None:
        self.lookback = lookback
        self.annualize = annualize
        self.dd_penalty = dd_penalty

    def score_series(self, price_series: pd.Series) -> tuple[float, float, float]:
        """Return (sharpe, max_dd, score) for a single price series."""
        ret = price_series.pct_change()
        mom = price_series / price_series.shift(self.lookback) - 1
        signal = (mom > 0).astype(int)
        strat_ret = signal.shift(1) * ret

        std = strat_ret.std()
        sharpe = (
            (strat_ret.mean() / std * np.sqrt(self.annualize))
            if std and std > 0
            else 0.0
        )

        cum = (1 + strat_ret).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1).min()

        score = sharpe - self.dd_penalty * abs(dd)
        return float(sharpe), float(dd), float(score)

    def score_dataframe(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """Score each column (ticker) and return a DataFrame with columns: ticker, sharpe, max_dd, score."""
        rows = []
        for ticker in close_df.columns:
            sharpe, dd, score = self.score_series(close_df[ticker])
            rows.append(
                {"ticker": ticker, "sharpe": sharpe, "max_dd": dd, "score": score}
            )
        score_df = pd.DataFrame(rows)
        return score_df.sort_values("score", ascending=False).reset_index(drop=True)


class UniverseSelector:
    """Selects a sector-balanced universe from a scored ticker DataFrame."""

    def __init__(
        self,
        ticker_to_sector: dict[str, str],
        max_per_sector: int = 8,
        target_size: int = 50,
    ) -> None:
        self.ticker_to_sector = ticker_to_sector
        self.max_per_sector = max_per_sector
        self.target_size = target_size

    def select(self, score_df: pd.DataFrame) -> list[str]:
        """Return list of tickers (up to target_size) with at most max_per_sector per sector, by score order."""
        universe: list[str] = []
        sector_count: dict[str, int] = {}

        for _, row in score_df.iterrows():
            ticker = row["ticker"]
            sector = self.ticker_to_sector.get(ticker)
            if sector is None:
                continue
            if sector_count.get(sector, 0) >= self.max_per_sector:
                continue
            universe.append(ticker)
            sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(universe) >= self.target_size:
                break

        return universe


class SP500Universe:
    """
    Builds an S&P 500–style universe: download prices, score by momentum, select a sector-balanced subset.
    """

    def __init__(
        self,
        data_downloader,
        sector_map: Optional[SectorMap] = None,
        start: str = "2016-01-01",
        end: str = "2025-12-31",
        min_observations: int = 500,
        download_delay_range: tuple[int, int] = (3, 5),
    ) -> None:
        self.data_downloader = data_downloader
        self.sector_map = sector_map if sector_map is not None else SectorMap()
        self.start = start
        self.end = end
        self.min_observations = min_observations
        self.download_delay_range = download_delay_range

        self._close_df: Optional[pd.DataFrame] = None
        self._score_df: Optional[pd.DataFrame] = None
        self._universe: Optional[list[str]] = None
        self._scorer = MomentumScorer()
        self._selector: Optional[UniverseSelector] = None

    def download_prices(self, tickers: Optional[list[str]] = None) -> pd.DataFrame:
        """Download adjusted close for each ticker; return aligned DataFrame. No global state."""
        if tickers is None:
            tickers = self.sector_map.all_tickers()

        series_list: list[pd.Series] = []
        for t in tickers:
            time.sleep(random.randint(*self.download_delay_range))
            try:
                df = self.data_downloader.yahoo(t, self.start, self.end)
            except Exception as e:
                print(f"Error downloading data for {t}: {e}")
                continue
            if len(df) < self.min_observations:
                continue
            s = df["Adj_close"].rename(t)
            series_list.append(s)

        if not series_list:
            return pd.DataFrame()

        close_df = pd.concat(series_list, axis=1)
        close_df = close_df.sort_index()
        close_df = close_df.dropna(axis=1, how="any")
        self._close_df = close_df
        return close_df

    def build(
        self,
        universe_size: int = 50,
        max_per_sector: int = 8,
        tickers: Optional[list[str]] = None,
    ) -> list[str]:
        """Download (if needed), score, select; return the chosen universe of tickers."""
        if self._close_df is None or tickers is not None:
            self.download_prices(tickers)

        if self._close_df is None or self._close_df.empty:
            return []

        self._score_df = self._scorer.score_dataframe(self._close_df)
        self._selector = UniverseSelector(
            self.sector_map.ticker_to_sector,
            max_per_sector=max_per_sector,
            target_size=universe_size,
        )
        self._universe = self._selector.select(self._score_df)
        return self._universe

    @property
    def close_df(self) -> pd.DataFrame:
        if self._close_df is None:
            return pd.DataFrame()
        return self._close_df

    @property
    def score_df(self) -> pd.DataFrame:
        if self._score_df is None:
            return pd.DataFrame()
        return self._score_df

    @property
    def universe(self) -> list[str]:
        return list(self._universe) if self._universe else []

    def sector_distribution(self) -> Counter[str]:
        """Return count of selected tickers per sector."""
        t2s = self.sector_map.ticker_to_sector
        return Counter(t2s[t] for t in self.universe if t in t2s)
