from threading import local

from app.fetch.alpha_vantage import AlphaVantageDownloader
from app.fetch.yahoo import DataDownloader
from app.schemas.settings import settings

Downloader = AlphaVantageDownloader | DataDownloader

_downloaders = local()


def get_yahoo_downloader() -> DataDownloader:
    if not hasattr(_downloaders, "yahoo"):
        _downloaders.yahoo = DataDownloader()
    return _downloaders.yahoo


def get_downloader() -> Downloader:
    provider = settings.data_provider.strip().lower()
    if provider == "yahoo":
        return get_yahoo_downloader()
    if provider == "alpha_vantage":
        if not hasattr(_downloaders, "alpha_vantage"):
            _downloaders.alpha_vantage = AlphaVantageDownloader()
        return _downloaders.alpha_vantage
    raise ValueError(f"Unsupported DATA_PROVIDER: {settings.data_provider}")
