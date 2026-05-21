# Trading Rookie

Monorepo: FastAPI backend + Next.js frontend.

```
trading_rookie/
├── backend/     # API, SQLite cache, Yahoo fetch, indicators, notebooks
└── frontend/    # Next.js + Lightweight Charts (UI shell from visactor-next-template)
```

## Quick start

**API** — see [backend/README.md](backend/README.md):

```cmd
conda activate quant
cd backend
uvicorn app.main:app --reload
```

**Frontend**:

```cmd
cd frontend
npm install
npm run dev
```

- API docs: http://127.0.0.1:8000/docs
- App: http://localhost:3000

## Frontend

See [frontend/README.md](frontend/README.md).
