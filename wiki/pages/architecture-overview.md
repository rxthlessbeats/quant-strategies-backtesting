# Architecture Overview

## Claims
- `trading_rookie` is a two-tier system: FastAPI backend plus Next.js frontend.
- Backend bootstraps DB on startup and always invokes scheduler startup during app lifespan; scheduled jobs run only when `refresh_scheduler_enabled` and `refresh_symbols` are configured (disabled by default).
- Frontend uses a persistent shell (`SideNav` + `AppHeader`) and route group under `(trading)`.
- `/chart` is client-heavy and orchestrates multiple API domains.

## Evidence
- Backend app composition and lifespan: `backend/app/main.py`.
- Scheduler wiring and guard conditions: `backend/app/services/market_data_scheduler.py`, `backend/app/schemas/settings.py` (`refresh_scheduler_enabled=False` by default).
- DB/session config: `backend/app/db/database.py`.
- Frontend shell: `frontend/src/app/layout.tsx`, `frontend/src/components/nav/app-header.tsx`.
- Trading routes: `frontend/src/app/(trading)/page.tsx`, `frontend/src/app/(trading)/chart/page.tsx`, `frontend/src/app/(trading)/indicators/page.tsx`, `frontend/src/app/(trading)/health/page.tsx`.
- Chart orchestration: `frontend/src/components/trading/chart-workspace.tsx`.

## Open Questions
- Should dashboard template blocks remain first-class in wiki, or be treated as legacy scaffolding?
- Should scheduler refresh cadence and module groups get a dedicated canonical page?

## Links
- [backend-api-surface](backend-api-surface.md)
- [backend-data-and-services](backend-data-and-services.md)
- [frontend-routing-and-composition](frontend-routing-and-composition.md)
- [frontend-data-flow-and-state](frontend-data-flow-and-state.md)
- [runbook-local-dev](runbook-local-dev.md)
- [glossary](glossary.md)
- [lint-report-2026-07-01](lint-report-2026-07-01.md)
