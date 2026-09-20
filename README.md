# RookieTrader

Monorepo: FastAPI backend + Next.js frontend.

```
trading_rookie/
├── backend/     # API, SQLite cache, Yahoo fetch, indicators, notebooks
└── frontend/    # Next.js + Lightweight Charts (UI shell from visactor-next-template)
```

## Quick start

Python is managed with [uv](https://docs.astral.sh/uv/) (lockfile at the repo root, interpreter from mise). Node stays on the existing `frontend/package-lock.json` — uv does not install npm packages.

```sh
uv sync
uv run start
```

- App: http://localhost:3000
- API docs: http://127.0.0.1:8000/docs

Node 20 is pinned in `.mise.toml`. If `npm` is missing: `mise install`.

## Deploy (Vercel)

Use **two Vercel projects** from this repo (`main` = production, `dev` = preview):

| Project | Root Directory | Env |
| --- | --- | --- |
| Frontend (Next.js) | `frontend` | `NEXT_PUBLIC_API_URL` = backend project URL |
| Backend (FastAPI) | `backend` | optional `DATA_PROVIDER`, `ALPHA_VANTAGE_API_KEY`, `DATABASE_URL` |

The API is a Vercel Function (`backend/app/main.py`). Default SQLite on Vercel lives in `/tmp` (cache only; it is not shared across instances). Yahoo/Alpha Vantage still fill data on request. In-process APScheduler does not run on Vercel.

For a durable cache later, set `DATABASE_URL` to Postgres (or similar) instead of SQLite.

## Frontend

See [frontend/README.md](frontend/README.md).
