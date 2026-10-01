# RookieTrader Backend

FastAPI service: SQLite cache → Yahoo fetch → technical indicators → JSON for K-line charts.

## Layout (single tree, no duplicates)

```
backend/
├── app/
│   ├── api/             # FastAPI routes
│   ├── schemas/         # Pydantic models
│   ├── services/        # Business logic
│   ├── db/              # SQLAlchemy + crud
│   ├── fetch/           # DataDownloader (Yahoo / Stooq)
│   ├── indicators/      # Technical indicators
│   └── research/        # SECTOR_MAP, universe (notebooks)
│       ├── const.py
│       └── universe/
├── notebooks/           # Jupyter + outputs/*.csv
├── data/                # stock_data.db (gitignored)
├── requirements.txt
└── Dockerfile
```

## Setup (uv)

From the repo root (not this folder):

```sh
uv sync
```

`requirements.txt` is an export of the lockfile for Docker.

## Run API

Whole stack from repo root:

```sh
uv run start
```

API only:

```sh
uv run --project . --directory backend python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/api/v1/stocks/{symbol}/bars` | OHLCV K-line bars |
| GET | `/api/v1/analysis/chart` | Bars + indicators |
| GET | `/api/v1/analysis/indicators` | Indicator catalog |

Example:

```
GET /api/v1/analysis/chart?symbol=AAPL&start=2024-01-01&end=2024-12-31&interval=1d&indicators=sma:5,sma:20,ema:50
```

## Indicator library

The catalog exposes 21 indicators and their default parameters:

| Category | Indicators |
|----------|------------|
| Trend | SMA, EMA, WMA, DEMA, TEMA, VWMA |
| Momentum | Momentum, RSI, MACD, ROC, Stochastic, Williams %R, CCI |
| Volatility | Bollinger Bands, ATR, NATR, Donchian channels |
| Volume | OBV, MFI, CMF, accumulation/distribution |

Use a bare ID for its defaults (`obv,atr,macd`), a single period (`wma:20`),
or named parameters (`stoch:period=14;smooth_k=3;smooth_d=3`). Periods and
smoothing windows must be whole numbers from 1 to 500; Bollinger standard
deviations may be fractional. MACD requires `fast < slow`.

Warm-up values are returned as `null`. EMA, DEMA, and TEMA start from the
first close; ATR uses a simple-average seed followed by Wilder smoothing.
Indicators use the supplied OHLCV history without future bars.

Run the formula and validation check from the repository root:

```sh
.venv/bin/python scripts/check-indicators.py
```

## Database

Default: `backend/data/stock_data.db`

```cmd
set DATABASE_URL=postgresql://user:pass@host:5432/stock_data
```

## Data Provider

Default provider: `yahoo`. Alpha Vantage remains supported:

```cmd
set DATA_PROVIDER=alpha_vantage
set ALPHA_VANTAGE_API_KEY=your_key_here
```

To use Yahoo:

```cmd
set DATA_PROVIDER=yahoo
```

## Cache and refresh

SQLite stores price history and market data modules. Charts and indicators share
one history frame per request. Price history has its own lock, separate from
fundamentals, and each worker reuses its Yahoo connection and authentication crumb.

`BAR_REFRESH_SECONDS=60` controls how often active daily bars are checked. Refreshes
include the latest session so its price can change. Completed sessions are reused
until the next trading day. The calendar uses weekdays; holidays are checked at
the short refresh interval. Successful empty responses also update the check time.
Historical date ranges are reused only when the requested range was fetched.

The frontend proxy gives successful responses a private 60-second browser cache.
Health checks, errors, and requests with `force` bypass it. Backend and proxy JSON
responses larger than 1 KB support gzip. No additional cache service or dependency
is required.

Run the isolated regression check from the repository root:

```sh
.venv/bin/python scripts/check-backend-cache.py
```

## Notebooks

```cmd
cd backend
jupyter notebook notebooks/mom_daily.ipynb
```

First cell uses `import _backend_path` then:

```python
from app.fetch.yahoo import DataDownloader
from app.research.const import SECTOR_MAP
from app.research.universe import SP500Universe
```

CSV outputs stay in `notebooks/outputs/`.

## Docker

```cmd
cd backend
docker build -t rookie-trader-api .
docker run -p 8000:8000 rookie-trader-api
```
