# Backend API Surface

## Claims
- Backend exposes `/health` and three `/api/v1` domains: `stocks`, `analysis`, `market`.
- Query parsing/normalization for chart endpoints is centralized in dependency functions plus Pydantic models.
- Error strategy primarily maps service failures to HTTP 400 or 502.

## Evidence
- Router registration: `backend/app/main.py`.
- Stocks routes: `backend/app/api/stock_routes.py` (`GET /api/v1/stocks/{symbol}/bars`).
- Analysis routes: `backend/app/api/analysis_routes.py`
  - `GET /api/v1/analysis/chart` — OHLCV plus computed indicator series (`indicators` query param, e.g. `sma:20,rsi:14`)
  - `GET /api/v1/analysis/indicators` — registry-backed indicator catalog
- Frontend analysis client: `frontend/src/lib/api.ts` (`fetchChart`, `fetchIndicatorCatalog`)
- Market routes: `backend/app/api/market_routes.py` (search, overview, areas/modules, metrics, benchmarks, performance).
- Query dependencies: `backend/app/api/deps.py`.
- Request models: `backend/app/schemas/requests.py`.

## Open Questions
- Should wiki include canonical response examples per endpoint family?
- Should API errors be normalized further across all routers?

## Links
- [architecture-overview](architecture-overview.md)
- [backend-data-and-services](backend-data-and-services.md)
- [frontend-data-flow-and-state](frontend-data-flow-and-state.md)
- [glossary](glossary.md)
