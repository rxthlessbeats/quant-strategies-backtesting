# Runbook Local Dev

## Claims
- Canonical local start is `uv run start` from the repo root (API + Next.js).
- Backend-only command is `uv run --project . --directory backend python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.
- Frontend-only command is `npm run dev` from `frontend/`.
- Python deps are locked in root `uv.lock` / `pyproject.toml`; Node deps stay on `frontend/package-lock.json`.
- Frontend API base defaults to `http://127.0.0.1:8000` when env var is absent.
- Backend defaults to SQLite DB at `backend/data/stock_data.db` when `DATABASE_URL` is unset.
- Alpha Vantage mode requires API key; provider choice is env-configured.
- Vercel can host both apps as two projects: Next.js Root Directory `frontend/`, FastAPI Root Directory `backend/` (`main` = production, `dev` = preview).
- On Vercel, SQLite defaults to `/tmp` (ephemeral cache). APScheduler is skipped. Persist with `DATABASE_URL` if needed.

## Evidence
- Frontend scripts: `frontend/package.json`.
- Frontend API base handling: `frontend/src/lib/api.ts`, `frontend/.env.example`.
- Backend app entrypoint: `backend/app/main.py`.
- Backend settings/env: `backend/app/schemas/settings.py`.
- Provider switch and key requirement: `backend/app/fetch/downloader.py`, `backend/app/fetch/alpha_vantage.py`.
- Backend dependencies: `pyproject.toml`, `uv.lock`; `backend/requirements.txt` is the Docker export.
- Dev launcher: `src/trading_rookie/start.py` (`uv run start`).
- Vercel frontend config: `frontend/vercel.json`.

## Open Questions
- Should team standardize provider/env profile for reproducible local dev?
- Should startup script(s) be canonicalized in repo to avoid terminal drift?

## Links
- [architecture-overview](architecture-overview.md)
- [backend-data-and-services](backend-data-and-services.md)
- [frontend-routing-and-composition](frontend-routing-and-composition.md)
