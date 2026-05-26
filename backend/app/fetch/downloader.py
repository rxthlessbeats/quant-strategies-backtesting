from app.fetch.alpha_vantage import AlphaVantageDownloader
from app.fetch.yahoo import DataDownloader
from app.schemas.settings import settings

Downloader = AlphaVantageDownloader | DataDownloader

_downloader: Downloader | None = None


def get_downloader() -> Downloader:
    global _downloader
    if _downloader is None:
        provider = settings.data_provider.strip().lower()
        if provider == "alpha_vantage":
            _downloader = AlphaVantageDownloader()
        elif provider == "yahoo":
            _downloader = DataDownloader()
        else:
            raise ValueError(f"Unsupported DATA_PROVIDER: {settings.data_provider}")
    return _downloader
