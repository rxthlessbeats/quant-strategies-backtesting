# Frontend Data Flow and State

## Claims
- `chart-workspace.tsx` is the main state orchestrator for symbol, indicators, view, panel datasets, and async fetch states.
- API access is centralized in `src/lib/api.ts` with typed response contracts in `src/lib/types.ts`.
- Indicator selections are normalized/serialized by `indicator-utils.ts`, including duplicate guards and query-string compatibility.
- Indicator catalog is loaded via `fetchIndicatorCatalog()` for toolbar and `/indicators` page; chart data uses `fetchChart()` with a serialized `indicators` query param.
- `chart-workspace.tsx` debounces chart fetches (~350ms), syncs indicator state to URL query params and localStorage presets.
- Chart rendering uses lightweight-charts and memoized transforms in `TradingChart.tsx`.
- Multiple panel components consume market and analysis payloads originating from chart workspace fetches.

## Evidence
- Main orchestrator: `frontend/src/components/trading/chart-workspace.tsx`.
- API client wrappers: `frontend/src/lib/api.ts`.
- Shared type contracts: `frontend/src/lib/types.ts`.
- Indicator selection logic: `frontend/src/lib/indicator-utils.ts` (`buildIndicatorsQuery`, `parseIndicatorSelections`).
- Catalog fetch usage: `frontend/src/components/trading/chart-toolbar.tsx`, `frontend/src/app/(trading)/indicators/page.tsx`.
- Chart fetch orchestration: `frontend/src/components/trading/chart-workspace.tsx` (`fetchChart` with debounced effect).
- Backend computation: `backend/app/services/indicator_service.py`, registry in `backend/app/indicators/registry.py`.
- Chart renderer: `frontend/src/components/trading/TradingChart.tsx`.
- Panel consumers: `frontend/src/components/trading/market-statistics-panel.tsx`, `performance-comparison-panel.tsx`, `analyst-recommendations-panel.tsx`, `company-stats-panel.tsx`.

## Open Questions
- Should fetch logic be split into dedicated hooks for testing and reuse?
- Which panels should be lazy-loaded to reduce initial chart route cost?

## Links
- [frontend-routing-and-composition](frontend-routing-and-composition.md)
- [backend-api-surface](backend-api-surface.md)
- [glossary](glossary.md)
