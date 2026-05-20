from app.fetch.yahoo import DataDownloader

_downloader: DataDownloader | None = None


def get_downloader() -> DataDownloader:
    global _downloader
    if _downloader is None:
        _downloader = DataDownloader()
    return _downloader
