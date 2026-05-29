from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _default_database_url() -> str:
    data_dir = _BACKEND_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'stock_data.db').as_posix()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default_factory=_default_database_url)
    backend_root: Path = Field(default=_BACKEND_ROOT)
    data_provider: str = "yahoo"
    alpha_vantage_api_key: str | None = None
    refresh_scheduler_enabled: bool = False
    refresh_symbol_universe: str = ""
    refresh_timezone: str = "America/New_York"
    refresh_market_close_hour: int = 16
    refresh_market_close_minute: int = 0
    earnings_refresh_delay_hours: int = 3

    @property
    def refresh_symbols(self) -> list[str]:
        return [
            symbol.strip().upper()
            for symbol in self.refresh_symbol_universe.split(",")
            if symbol.strip()
        ]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
