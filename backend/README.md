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

## Setup (Mamba / conda)

```cmd
conda activate quant
cd backend

mamba install -y fastapi uvicorn sqlalchemy pydantic pydantic-settings pandas numpy requests pandas-datareader
pip install -r requirements.txt
```

## Run API

```cmd
cd backend
uvicorn app.main:app --reload
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

## Database

Default: `backend/data/stock_data.db`

```cmd
set DATABASE_URL=postgresql://user:pass@host:5432/stock_data
```

## Data Provider

Default provider: `alpha_vantage`

```cmd
set DATA_PROVIDER=alpha_vantage
set ALPHA_VANTAGE_API_KEY=your_key_here
```

To use the existing Yahoo downloader instead:

```cmd
set DATA_PROVIDER=yahoo
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
