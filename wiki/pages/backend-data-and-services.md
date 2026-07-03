# Backend Data and Services

## Claims
- SQLite is default DB with WAL and process-level write lock semantics.
- Core persistence models are bars, fetch metadata, cached market modules, and company fundamentals.
- Provider selection is dynamic (`yahoo` or `alpha_vantage`) via settings.
- Market module refresh policy depends on module class and earnings timing heuristics.
- Indicator computation is registry-driven and parameterized from query specs.

## Evidence
- DB config and lock: `backend/app/db/database.py`.
- ORM models: `backend/app/db/models.py`.
- CRUD operations: `backend/app/db/crud.py`.
- Sync and OHLCV retrieval: `backend/app/services/sync_service.py`, `backend/app/services/stock_data_service.py`.
- Downloader selection: `backend/app/fetch/downloader.py`.
- Yahoo provider: `backend/app/fetch/yahoo.py`.
- Alpha Vantage provider: `backend/app/fetch/alpha_vantage.py`.
- Market module orchestration: `backend/app/services/market_data_service.py`.
- Scheduler jobs: `backend/app/services/market_data_scheduler.py`.
- Indicator service: `backend/app/services/indicator_service.py`.

## Open Questions
- Which provider should be default for team/dev ergonomics?
- Should fundamentals remain flattened strings or migrate to typed numeric storage?

## Links
- [backend-api-surface](backend-api-surface.md)
- [architecture-overview](architecture-overview.md)
- [runbook-local-dev](runbook-local-dev.md)
