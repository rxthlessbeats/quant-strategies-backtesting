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
| Frontend (Next.js) | `frontend` | `API_URL` = backend project URL; `API_SECRET` = same random string as backend |
| Backend (FastAPI) | `backend` | `API_SECRET` (required); optional `DATA_PROVIDER`, `ALPHA_VANTAGE_API_KEY`, `DATABASE_URL` |

Do **not** use `NEXT_PUBLIC_` for the secret. The browser only calls same-origin `/api/backend/...`; Next.js attaches `X-API-Key` server-side. Direct calls to the FastAPI URL without that header get 401.

The API is a Vercel Function (`backend/app/main.py`). Default SQLite on Vercel lives in `/tmp` (cache only; it is not shared across instances). Yahoo still fills data on request. In-process APScheduler does not run on Vercel.

For a durable cache later, set `DATABASE_URL` to Postgres (or similar) instead of SQLite.

## Frontend

See [frontend/README.md](frontend/README.md).
