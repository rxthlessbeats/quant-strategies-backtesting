# Runbook Local Dev

## Claims
- Backend dev command is `uvicorn app.main:app --reload` from `backend/`.
- Frontend dev command is `npm run dev` from `frontend/`.
- Frontend API base defaults to `http://127.0.0.1:8000` when env var is absent.
- Backend defaults to SQLite DB at `backend/data/stock_data.db` when `DATABASE_URL` is unset.
- Alpha Vantage mode requires API key; provider choice is env-configured.

## Evidence
- Frontend scripts: `frontend/package.json`.
- Frontend API base handling: `frontend/src/lib/api.ts`, `frontend/.env.example`.
- Backend app entrypoint: `backend/app/main.py`.
- Backend settings/env: `backend/app/schemas/settings.py`.
- Provider switch and key requirement: `backend/app/fetch/downloader.py`, `backend/app/fetch/alpha_vantage.py`.
- Backend dependencies: `backend/requirements.txt`.

## Open Questions
- Should team standardize provider/env profile for reproducible local dev?
- Should startup script(s) be canonicalized in repo to avoid terminal drift?

## Links
- [architecture-overview](architecture-overview.md)
- [backend-data-and-services](backend-data-and-services.md)
- [frontend-routing-and-composition](frontend-routing-and-composition.md)
